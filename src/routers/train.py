# src/routers/train.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from services.model_manager import ModelManager
from db.session import SessionLocal
from db.models import TrainingRun
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from sqlalchemy.orm import Session
from db.models import Dataset, TrainingRun
from db.session import SessionLocal
import time
import mlflow

router = APIRouter(
    prefix="/train",
    tags=["Train"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=TrainResponse)
def train_endpoint(request: TrainRequest, db: Session = Depends(get_db)):
    """Endpoint pour entraîner le modèle NER."""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="train_create").inc()
    try:
        # Charger le modèle
        load_model()

        # Récupérer le dataset depuis la DB
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Entraîner le modèle
        train_model(train_data=dataset.data)

        # Enregistrer le run dans la DB
        training_run = TrainingRun(
            run_id="mlflow_run_id_placeholder",  # Remplacer par le vrai run_id après intégration MLflow
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        db.add(training_run)
        db.commit()
        db.refresh(training_run)
        
        return TrainResponse(
            id=training_run.id,
            run_id=training_run.run_id,
            dataset_id=training_run.dataset_id,
            epochs=training_run.epochs,
            batch_size=training_run.batch_size,
            status=training_run.status,
            created_at=training_run.created_at.isoformat()
        )
    except Exception as e:
        # Enregistrer le run en échec
        training_run = TrainingRun(
            run_id="",
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status=f"Failed: {str(e)}"
        )
        db.add(training_run)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="train_create").observe(latency)
