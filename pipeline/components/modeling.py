import kfp
from kfp import dsl

@dsl.component(
    base_image="microwave1005/scipy-img:latest",
    packages_to_install=[
        "mlflow", "optuna", "shap", "loguru", "lightgbm", "xgboost", "joblib"
    ],
)
def modeling_component(
    processed_train: dsl.Input[dsl.Dataset],
    processed_test:  dsl.Input[dsl.Dataset],
    registered_uri:  dsl.Output[dsl.Artifact],
    model_joblib:    dsl.Output[dsl.Artifact],

    model_name:      str = "xgb",    # "xgb" | "lgbm"
    version:         str = "v1",
    experiment_name: str = "demo-kubeflow",

    mlflow_endpoint_url:     str = "http://minio.minio.svc.cluster.local:9000",
    mlflow_access_key_id:    str = "minio",
    mlflow_secret_access_key:str = "minio123",
    mlflow_tracking_uri:     str = "http://mlflow.mlflow.svc.cluster.local:5000",
):
    """
    Train an XGBoost / LightGBM model with Optuna tuning on a held-out split,
    refit on full data, then log accuracy, SHAP plot, and model to MLflow.
    """
    import os
    import os.path as osp
    from tempfile import NamedTemporaryFile

    import joblib, optuna, mlflow, shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import xgboost as xgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from loguru import logger

    # ─── MLflow / MinIO setup ─────────────────────────────────────────────────
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"]      = mlflow_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"]  = mlflow_secret_access_key
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment = {}", experiment_name)

    # ─── load full train data ───────────────────────────────────────────────────
    df = pd.read_csv(processed_train.path)
    y_full = df["TARGET"].values
    X_full = df.drop(columns=["TARGET"])

    # ─── split off a validation set for Optuna ────────────────────────────────
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # ─── Optuna hyper-parameter tuning ────────────────────────────────────────
    def suggest(trial):
        return {
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    def objective_xgb(trial):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **suggest(trial))
        m.fit(X_tune, y_tune)
        preds = m.predict(X_val)
        return accuracy_score(y_val, preds)

    def objective_lgbm(trial):
        m = LGBMClassifier(**suggest(trial))
        m.fit(X_tune, y_tune)
        preds = m.predict(X_val)
        return accuracy_score(y_val, preds)

    logger.info("Running Optuna tuning")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective_xgb if model_name.lower() == "xgb" else objective_lgbm,
        n_trials=10,
        show_progress_bar=False,
    )
    best_params = study.best_params
    logger.success("Best params: {}", best_params)

    # ─── final fit on full data ────────────────────────────────────────────────
    model = (
        xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", **best_params)
        if model_name.lower() == "xgb"
        else LGBMClassifier(**best_params)
    )
    model.fit(X_full, y_full)

    # ─── prepare evaluation set ────────────────────────────────────────────────
    X_eval, y_eval = X_full, y_full
    # if an external test with labels exists, use it instead
    if processed_test.path and osp.exists(processed_test.path):
        df_test = pd.read_csv(processed_test.path)
        if "TARGET" in df_test.columns and not df_test.empty:
            y_eval = df_test["TARGET"].values
            X_eval = df_test.drop(columns=["TARGET"])

    # ─── compute accuracy ─────────────────────────────────────────────────────
    preds = model.predict(X_eval)
    acc   = accuracy_score(y_eval, preds)

    # ─── save SHAP summary plot ────────────────────────────────────────────────
    with NamedTemporaryFile(delete=False, suffix="_shap.png") as f:
        shap_values = shap.Explainer(model)(X_eval)
        shap.summary_plot(shap_values, X_eval, show=False)
        plt.gcf().savefig(f.name)
        shap_path = f.name
    plt.close()

    # ─── save model artifact ──────────────────────────────────────────────────
    joblib.dump(model, model_joblib.path)

    # ─── log to MLflow ────────────────────────────────────────────────────────
    run_tag  = "XGBoost" if model_name.lower() == "xgb" else "LightGBM"
    run_name = f"{version}_{run_tag}"

    with mlflow.start_run(run_name=f"{run_name}_run") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(shap_path,       artifact_path="model_artifacts")
        mlflow.log_artifact(model_joblib.path, artifact_path="model_artifacts")
        if model_name.lower() == "xgb":
            mlflow.xgboost.log_model(model, artifact_path="flavour")
        else:
            mlflow.lightgbm.log_model(model, artifact_path="flavour")
        uri = mlflow.get_artifact_uri("model_artifacts")
        mlflow.register_model(uri, run_name)

    # ─── write registered URI for downstream steps ───────────────────────────
    with open(registered_uri.path, "w") as f_out:
        f_out.write(uri)

    logger.success("Model training, logging, and registration complete")
