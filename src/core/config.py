import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/gliner_db")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", 8000))

    class Config:
        env_file = ".env"

settings = Settings()
