# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import os

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
    prometheus_port: int = 8008

    class Config:
        # Utilisation d'un chemin absolu pour garantir que le fichier .env est bien trouvé
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = 'utf-8'

# Instanciation de la configuration
settings = Settings()
