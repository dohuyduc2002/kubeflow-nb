# settings.py
from dataclasses import dataclass, field

@dataclass
class MLflowConfig:
    endpoint_url: str       = "http://minio.minio.svc.cluster.local:9000"
    access_key_id: str      = "minio"
    secret_access_key: str  = "minio123"
    tracking_uri: str       = "http://mlflow.mlflow.svc.cluster.local:5000"

@dataclass
class KubeflowConfig:
    username: str   = "user@example.com"
    password: str   = "12341234"
    namespace: str  = "kubeflow-user-example-com"
    host: str       = "http://localhost:8080"

@dataclass
class Settings:
    mlflow:   MLflowConfig   = field(default_factory=MLflowConfig)
    kubeflow: KubeflowConfig = field(default_factory=KubeflowConfig)

settings = Settings()
