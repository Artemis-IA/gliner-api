# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import torch


class Settings(BaseSettings):
    
    # Proxy settings
    use_et_proxy: bool = False
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = ""

    # PostgreSQL
    pg_major: int
    postgre_port: int
    postgre_user: str
    postgre_password: str
    postgre_db: str
    postgre_host: str
    database_url: str
    django_db: str
    
    # MLflow
    mlflow_user: str
    mlflow_password: str
    mlflow_db: str
    mlflow_backend_store_uri: str
    mlflow_artifact_root: str
    mlflow_port: int
    mlflow_tracking_uri: str
    mlflow_s3_endpoint_url: str
    mlflow_s3_ignore_tls: bool

    # MinIO
    minio_port: int
    minio_console_port: int
    minio_client_port: int
    minio_access_key: str
    minio_secret_key: str
    minio_root_user: str
    minio_root_password: str
    minio_api_url: str

    # AWS Credentials for MLflow (MinIO compatible)
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str

    # Label Studio
    label_studio_user: str
    label_studio_password: str
    label_studio_db: str
    label_studio_host: str
    label_studio_port: int
    label_studio_username: str
    label_studio_email: str
    label_studio_api_key: str
    label_studio_bucket_name: str
    label_studio_bucket_prefix: str
    label_studio_bucket_endpoint_url: str
    label_studio_bucket_access_key: str
    label_studio_bucket_secret_key: str
    label_studio_target_bucket: str
    label_studio_target_prefix: str
    label_studio_target_access_key: str
    label_studio_target_secret_key: str
    label_studio_target_endpoint_url: str
    label_studio_project_name: str
    label_studio_project_title: str
    label_studio_local_files_serving_enabled: bool
    ls_database_url: str
    django_db: str
    # Secret Key
    secret_key: str

    # ML Backend Configuration
    mlbackend_port: int
    label_studio_ml_backends: str
    gliner_basic_auth_user: str
    gliner_basic_auth_pass: str
    gliner_model_name: str

    # Other configurations
    workers: int
    threads: int
    test_env: str
    locip: str

    # Prometheus
    prometheus_port: int = 8008

    # Models
    default_models: str = "urchade/gliner_smallv2.1"

    train_config: dict = {
        "num_steps": 10_000,
        "train_batch_size": 2,
        "eval_every": 1_000,
        "save_directory": "checkpoints",
        "warmup_ratio": 0.1,
        "device": "cuda",
        "lr_encoder": 1e-5,
        "lr_others": 5e-5,
        "freeze_token_rep": False,
        "max_types": 25,
        "shuffle_types": True,
        "random_drop": True,
        "max_neg_type_ratio": 1,
        "max_len": 384,
    }

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = 'utf-8'


# Instanciation de la configuration
settings = Settings()

MODELS = {
    "GLiNER-S": "urchade/gliner_smallv2.1",
    "GLiNER-M": "urchade/gliner_mediumv2.1",
    "GLiNER-L": "urchade/gliner_largev2.1",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CONFIG = {
    "num_steps": 10_000,
    "train_batch_size": 2,
    "eval_every": 1_000,
    "save_directory": "checkpoints",
    "warmup_ratio": 0.1,
    "device": DEVICE,
    "lr_encoder": 1e-5,
    "lr_others": 5e-5,
    "freeze_token_rep": False,
    "max_types": 25,
    "shuffle_types": True,
    "random_drop": True,
    "max_neg_type_ratio": 1,
    "max_len": 384,
}
