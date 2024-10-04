# tests/utils/test_mlflow_setup.py
from src.utils.mlflow_setup import setup_mlflow
import mlflow

def test_setup_mlflow(monkeypatch):
    monkeypatch.setattr("src.core.config.settings.mlflow_tracking_uri", "http://localhost:5000")
    setup_mlflow()
    assert mlflow.get_tracking_uri() == "http://localhost:5000"
