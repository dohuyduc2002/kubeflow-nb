import argparse
import datetime as dt
from pathlib import Path
import kfp


DEFAULT_YAML = "credit_underwriting_pipeline.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run-underwriting",
        description="Submit credit-underwriting pipeline YAML to Kubeflow (KFP v2)",
    )

    # ▸ KFP endpoint
    p.add_argument("--host",
                   help="KFP endpoint when running outside the cluster")

    # ▸ MinIO / S3
    p.add_argument("--endpoint_url",
                   default="http://minio.minio.svc.cluster.local:9000")
    p.add_argument("--bucket_name",  default="sample-data")
    p.add_argument("--train_key",    default="data/application_train.csv")
    p.add_argument("--test_key",     default="data/application_test.csv")
    p.add_argument("--aws_access_key_id",     default="minio")
    p.add_argument("--aws_secret_access_key", default="minio123")

    # ▸ MLflow tracking
    p.add_argument("--mlflow_tracking_uri",
                   default="http://mlflow.mlflow.svc.cluster.local:5000")

    # ▸ Experiment settings
    p.add_argument("--target_col",     default="TARGET")
    p.add_argument("--model_name",     choices=["xgb", "lgbm"], default="lgbm")
    p.add_argument("--version",        default="v1")
    p.add_argument("--experiment_name", default="demo-kubeflow")

    # ▸ Misc
    p.add_argument("--yaml_path", default=DEFAULT_YAML,
                   help=f"Compiled pipeline YAML (default: {DEFAULT_YAML})")
    p.add_argument("--run_name",
                   help="Optional run name (defaults to timestamp)")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    yaml_path = Path(args.yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).resolve().parent / yaml_path
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pipeline YAML not found: {yaml_path}")

    client = kfp.Client(host=args.host) if args.host else kfp.Client()

    params = {
        "endpoint_url":           args.endpoint_url,
        "bucket_name":            args.bucket_name,
        "train_key":              args.train_key,
        "test_key":               args.test_key,
        "aws_access_key_id":      args.aws_access_key_id,
        "aws_secret_access_key":  args.aws_secret_access_key,
        "mlflow_tracking_uri":    args.mlflow_tracking_uri,
        "target_col":             args.target_col,
        "model_name":             args.model_name,
        "version":                args.version,
        "experiment_name":        args.experiment_name,
    }

    run_name = args.run_name or f"underwriting-{dt.datetime.now():%Y%m%d-%H%M%S}"

    run = client.create_run_from_pipeline_package(
        pipeline_file=str(yaml_path),
        arguments=params,
        experiment_name=args.experiment_name,
        run_name=run_name,
    )

    print(f" Pipeline submitted!  Run ID: {run.run_id}")


if __name__ == "__main__":
    main()
