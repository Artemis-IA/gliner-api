# src/utils/mlflow_setup.py
import mlflow
from core.config import settings

def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("GLiNER_Experiments")
