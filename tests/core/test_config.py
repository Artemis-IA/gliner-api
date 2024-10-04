# tests/core/test_config.py
from src.core.config import settings
import os

def test_config_variables():
    assert settings.database_url is not None
    assert settings.mlflow_tracking_uri is not None
    assert settings.prometheus_port == 8008
    assert settings.postgres_user is not None
    assert settings.postgres_password is not None
    assert settings.postgres_db is not None
    assert settings.postgres_port is not None
    assert settings.postgres_host is not None
    assert settings.mlflow_port is not None
    assert settings.mlflow_backend_store_uri is not None
    assert settings.mlflow_artifact_root is not None
    assert os.path.exists(settings.env_file)
    assert settings.env_file_encoding == 'utf-8'
    