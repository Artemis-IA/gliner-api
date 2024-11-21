# utils/mlflow_service.py
import os
import time
import mlflow
from typing import Dict, Any, Optional
from fastapi import HTTPException
from utils.device_metrics import DeviceMetricsTracker
from utils.huggingface_helper import HuggingFaceHelper
from mlflow.tracking import MlflowClient
from loguru import logger
from core.config import settings

class MLFlowService:
    def __init__(self):
        self.mlflow_experiment_name = "doc_processing"
        self.mlflow_run_name = None
        self.mlflow_tracking_uri = settings.mlflow_tracking_uri
        self.mlflow_artifact_location = settings.mlflow_artifact_root
        self.client = MlflowClient()
        self.device_tracker = DeviceMetricsTracker(project_name="doc_processing")
        self.hf_helper = HuggingFaceHelper()
        self.huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
        self.models = {}
        self.static_models = {
            "Ollama Embedding Model": (
                "sentence-transformers/all-MiniLM-L6-v2",
                os.path.join(self.huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2"),
            ),
            "GLiNER Extractor Model": (
                "E3-JSI/gliner-multi-pii-domains-v1",
                os.path.join(self.huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1"),
            ),
            "Gliner Transformer Model": (
                "knowledgator/gliner-multitask-large-v0.5",
                os.path.join(self.huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5"),
            ),
            "Tokenizer Model": (
                "microsoft/deberta-v3-large",
                os.path.join(self.huggingface_cache, "models--microsoft--deberta-v3-large"),
            ),
        }

    def log_params(self, params: Dict[str, Any]):
        """Log parameters in the current MLflow run."""
        for param, value in params.items():
            mlflow.log_param(param, value)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics in the current MLflow run."""
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

    def log_model_details(self) -> Dict[str, str]:
        """Log all static models to MLflow."""
        logger.info("Starting model logging process...")
        try:
            with mlflow.start_run(run_name="Model Logging"):
                for model_name, (model_id, model_file_path) in self.static_models.items():
                    logger.info(f"Processing model: {model_name}")
                    metadata = self.hf_helper.fetch_model_metadata(model_id)
                    self._log_model_metadata(model_name, model_file_path, metadata)
            return {"message": "All static models logged successfully."}
        except Exception as e:
            logger.error(f"Error during model logging: {e}")
            raise Exception("Error logging models.")

    def _log_model_metadata(self, model_name: str, model_file_path: str, metadata: dict, run_id: str = None):
        """Log metadata and artifact for a specific model."""
        try:
            # Check and register model in MLflow
            registered_models = [rm.name for rm in self.client.search_registered_models()]
            if model_name not in registered_models:
                self.client.create_registered_model(model_name)

            # Log metadata
            mlflow.set_tag(f"{model_name}_description", metadata.get("description", ""))
            mlflow.log_param(f"{model_name}_version", metadata.get("version", "unknown"))

            # Log model artifact
            if os.path.exists(model_file_path):
                artifact_path = f"artifacts/{model_name}"
                mlflow.log_artifact(model_file_path, artifact_path=artifact_path)
                self.client.create_model_version(
                    name=model_name,
                    source=f"{mlflow.get_artifact_uri()}/{artifact_path}",
                    run_id=run_id,
                )
            else:
                logger.warning(f"Model file path not found: {model_file_path}")
        except Exception as e:
            logger.error(f"Error logging metadata for model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error logging model metadata: {e}")

    def log_query(self, query: str) -> Dict[str, str]:
        """Log a query in MLflow."""
        try:
            with mlflow.start_run(run_name="Query Logging") as run:
                mlflow.log_param("query", query)
                mlflow.log_param("timestamp", time.time())
                logger.info(f"Query logged: {query}")
                return {"message": "Query logged successfully."}
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            raise HTTPException(status_code=500, detail="Error logging query.")
