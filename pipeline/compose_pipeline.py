import kfp
from kfp import dsl, compiler

# Adjust to your package structure
from components.dataloader           import dataloader_component
from components.detect_feature_types import detect_feature_types_component
from components.preprocess_binning   import preprocess_binning_component
from components.modeling             import modeling_component

# ───────────────────────── pipeline ───────────────────────────────
@dsl.pipeline(
    name="credit-underwriting-training-pipeline",
    description="End-to-end underwriting flow (MinIO ➜ preprocess ➜ train ➜ MLflow)"
)
def credit_underwriting_pipeline(
    # ► MinIO / S3 backend (defaults point to in-cluster services)
    endpoint_url: str = "http://minio.minio.svc.cluster.local:9000",
    bucket_name : str = "sample-data",
    train_key   : str = "data/application_train.csv",
    test_key    : str = "data/application_test.csv",
    aws_access_key_id    : str = "minio",
    aws_secret_access_key: str = "minio123",

    # ► MLflow tracking (uses the same MinIO as artifact store)
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",

    # ► ML / experiment hyper-parameters
    target_col      : str = "TARGET",
    model_name      : str = "lgbm",     # "xgb" | "lgbm"
    version         : str = "v1",
    experiment_name : str = "demo-kubeflow",
):
    #  Download raw CSVs from MinIO
    load_step = dataloader_component(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        test_key=test_key,
        train_key=train_key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    #  Detect column types
    detect_step = detect_feature_types_component(
        train_csv=load_step.outputs["train_output"],
    )

    #  Preprocess → binning → feature selection
    preprocess_step = preprocess_binning_component(
        train_csv=load_step.outputs["train_output"],
        test_csv=load_step.outputs["test_output"],
        categorical_cols_json=detect_step.outputs["categorical_cols_json"],
        numerical_cols_json =detect_step.outputs["numerical_cols_json"],
        target_col=target_col,
        data_version=version,
    )

    #  Model training, Optuna tuning & MLflow logging
    modeling_component(
        processed_train=preprocess_step.outputs["processed_train"],
        processed_test =preprocess_step.outputs["processed_test"],
        model_name=model_name,
        version=version,
        experiment_name=experiment_name,

        # Pass MinIO / MLflow credentials straight in
        mlflow_endpoint_url=endpoint_url,          # same S3 backend
        mlflow_access_key_id=aws_access_key_id,
        mlflow_secret_access_key=aws_secret_access_key,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

# ────────────────────────── compile ───────────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=credit_underwriting_pipeline,
        package_path="credit_underwriting_pipeline.yaml",
    )
    print("✅ Pipeline compiled → credit_underwriting_pipeline.yaml")
