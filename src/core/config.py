# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Variables d'environnement
    database_url: str
    mlflow_tracking_uri: str
    mlflow_backend_store_uri: str
    mlflow_artifact_root: str
    mlflow_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_port: int
    postgres_host: str
    
    minio_port: int
    minio_root_user: str
    minio_root_password: str
    minio_api_url: str
    
    mlflow_s3_endpoint_url: str
    mlflow_s3_ignore_tls: bool
    
    prometheus_port: int = 8008
    
    default_model: str = "urchade/gliner_smallv2.1"
    train_config: dict = {
        "num_steps": 10_000,  # N training iteration
        "train_batch_size": 2,  # batch size for training
        "eval_every": 1_000,  # evaluation/saving steps
        "save_directory": "checkpoints",  # where to save checkpoints
        "warmup_ratio": 0.1,  # warmup steps
        "device": "cuda",  # placeholder, will be set dynamically
        "lr_encoder": 1e-5,  # learning rate for the backbone
        "lr_others": 5e-5,  # learning rate for other parameters
        "freeze_token_rep": False,  # freeze of not the backbone
        "max_types": 25,  # maximum number of entity types during training
        "shuffle_types": True,  # if shuffle or not entity types
        "random_drop": True,  # randomly drop entity types
        "max_neg_type_ratio": 1,  # ratio of positive/negative types
        "max_len": 384,  # maximum sentence length
    }

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = 'utf-8'

# Instanciation de la configuration
settings = Settings()
