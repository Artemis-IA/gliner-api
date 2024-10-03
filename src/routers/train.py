from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.train import TrainRequest, TrainResponse
from app.db.session import SessionLocal
from app.db.models import Dataset, TrainingRun
from app.utils.gliner_utils import train_gliner_model, create_gliner_dataset
from app.core.prometheus_setup import REQUEST_COUNT, REQUEST_LATENCY
from app.core.config import settings
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
def train_model(request: TrainRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="train_create").inc()
    try:
        # Récupérer le dataset depuis la DB
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_path = create_gliner_dataset(dataset.data, format="json-ner")
        
        # Démarrer un run MLflow
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("dataset_id", request.dataset_id)
            mlflow.log_param("epochs", request.epochs)
            mlflow.log_param("batch_size", request.batch_size)
            
            # Entraîner le modèle
            model_output = train_gliner_model(dataset_path, request.epochs, request.batch_size)
            
            # Log des résultats
            mlflow.log_artifact(model_output, "model")
        
        # Enregistrer le run dans la DB
        training_run = TrainingRun(
            run_id=run_id,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        db.add(training_run)
        db.commit()
        db.refresh(training_run)
        
        return TrainResponse(
            run_id=training_run.run_id,
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
