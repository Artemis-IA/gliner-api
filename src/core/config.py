# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    # Variables d'environnement
    database_url: str
    mlflow_tracking_uri: str
    mlflow_backend_store_uri: str
    mlflow_artifact_root: str
    mlflow_db: str
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
    
    default_models: str = "urchade/gliner_smallv2.1"

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