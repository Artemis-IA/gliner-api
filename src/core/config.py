# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    database_url: str
    mlflow_tracking_uri: str
    prometheus_port: int = 8008

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"  # Ajuster le chemin vers .env

settings = Settings()
