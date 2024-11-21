from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from utils.mlflow_service import MLFlowService
from loguru import logger

router = APIRouter()

mlflow_service = MLFlowService()


@router.post("/log_model")
async def log_model(model_name: str, model_path: str):
    """Log a single model to MLflow."""
    try:
        result = mlflow_service.log_model_details()
        return {"message": f"Model '{model_name}' logged successfully.", "details": result}
    except Exception as e:
        logger.error(f"Error logging model '{model_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error logging model '{model_name}'.")


@router.post("/log_static_models")
async def log_static_models():
    """Log all static models to MLflow."""
    try:
        result = mlflow_service.log_model_details()
        return {"message": "Static models logged successfully.", "details": result}
    except Exception as e:
        logger.error(f"Error logging static models: {e}")
        raise HTTPException(status_code=500, detail="Error logging static models.")


@router.post("/log_params")
async def log_params(params: Dict[str, Any]):
    """Log parameters in an active MLflow run."""
    try:
        mlflow_service.log_params(params)
        return {"message": "Parameters logged successfully."}
    except Exception as e:
        logger.error(f"Error logging parameters: {e}")
        raise HTTPException(status_code=500, detail="Error logging parameters.")


@router.post("/log_metrics")
async def log_metrics(metrics: Dict[str, Any]):
    """Log metrics in an active MLflow run."""
    try:
        mlflow_service.log_metrics(metrics)
        return {"message": "Metrics logged successfully."}
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise HTTPException(status_code=500, detail="Error logging metrics.")


@router.post("/log_query")
async def log_query(query: str):
    """Log a query to MLflow."""
    try:
        result = mlflow_service.log_query(query)
        return result
    except Exception as e:
        logger.error(f"Error logging query: {e}")
        raise HTTPException(status_code=500, detail="Error logging query.")


@router.get("/status")
async def get_mlflow_status():
    """Check MLflow status."""
    try:
        tracking_uri = mlflow_service.mlflow_tracking_uri
        return {"status": "OK", "tracking_uri": tracking_uri}
    except Exception as e:
        logger.error(f"Error fetching MLflow status: {e}")
        raise HTTPException(status_code=500, detail="Error fetching MLflow status.")
