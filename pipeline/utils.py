from typing import Dict, Optional
from urllib.parse import urlsplit, urlencode
import kfp
import requests
import urllib3

SCIPY_IMAGE = "microwave1005/scipy-img:latest"
# MinIO configuration
MINIO_ENDPOINT= "minio.minio.svc.cluster.local:9000"
MINIO_ACCESS_KEY= "minio"
MINIO_SECRET_KEY= "minio123"
MINIO_BUCKET_NAME= "sample-data"

MLFLOW_ENDPOINT= "mlflow.mlflow.svc.cluster.local:5000"
KFP_NAMESPACE='kubeflow-user-example-com'

def upload_pipeline(kfp_client, pipeline_yaml, pipeline_name, version_name):

    # Upload pipeline
    pipeline = kfp_client.upload_pipeline(
        pipeline_package_path=pipeline_yaml,
        pipeline_name=pipeline_name,
        namespace="kubeflow-user-example-com",  # Adjust namespace as needed
    )
    pipeline_id = getattr(pipeline, "pipeline_id")
    print(f"⬆️  Uploaded pipeline: {pipeline_name} (id={pipeline_id})")

    pipeline_version = kfp_client.upload_pipeline_version(
        pipeline_package_path=pipeline_yaml,
        pipeline_version_name=version_name,
        pipeline_id=pipeline_id,
    )
    version_id = getattr(pipeline_version, "pipeline_version_id")
    print(f"⬆️  Uploaded pipeline version: {version_name} (id={version_id})")
    
    return pipeline_id, version_id, version_name

