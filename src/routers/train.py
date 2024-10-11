# src/routers/train.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from models.ner_model import NERModel
from db.session import SessionLocal
from db.models import TrainingRun, Dataset
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from sqlalchemy.orm import Session
import time
import uuid
from loguru import logger

router = APIRouter(
    prefix="/train",
    tags=["Train"]
)

# Dependency pour obtenir la session DB
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
    run_id = str(uuid.uuid4())  # Génère un ID de run unique

    try:
        # Récupérer le dataset depuis la DB
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset non trouvé")

        # Vérifier que le dataset contient des données d'entraînement et d'évaluation
        train_data = dataset.train_data  # Assurez-vous que 'train_data' est une liste de dicts
        eval_data = dataset.eval_data  # Assurez-vous que 'eval_data' est un dict avec 'samples'

        if not train_data:
            raise HTTPException(status_code=400, detail="Données d'entraînement vides")
        if eval_data is None or 'samples' not in eval_data:
            raise HTTPException(status_code=400, detail="Données d'évaluation manquantes ou incomplètes")

        # Préparer la configuration d'entraînement
        train_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "save_directory": "checkpoints",  # Exemple, ajustez selon vos besoins
            "lr_others": 5e-5,  # Exemple, ajustez selon vos besoins
            "warmup_ratio": 0.1,
            "max_types": 25,
            "shuffle_types": True,
            "random_drop": True,
            "max_neg_type_ratio": 1,
            "max_len": 384,
            "eval_every": 100
        }

        # Initialiser le NERModel avec les paramètres d'entraînement
        ner_model = NERModel(
            name=dataset.model_name,  # Assurez-vous que le dataset a un champ 'model_name'
            train_config=train_config
        )

        # Entraîner le modèle
        ner_model.train(train_data=train_data, eval_data=eval_data)

        # Enregistrer le run d'entraînement dans la DB
        training_run = TrainingRun(
            run_id=run_id,  # Utiliser l'ID de run généré
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        db.add(training_run)
        db.commit()
        db.refresh(training_run)

        # Incrémenter les métriques Prometheus pour une réponse réussie
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status="200").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status="200").observe(time.time() - start_time)

        return TrainResponse(
            id=training_run.id,
            run_id=training_run.run_id,
            dataset_id=training_run.dataset_id,
            epochs=training_run.epochs,
            batch_size=training_run.batch_size,
            status=training_run.status,
            created_at=training_run.created_at.isoformat()
        )

    except HTTPException as http_exc:
        # Enregistrer un run d'entraînement échoué
        training_run = TrainingRun(
            run_id=run_id,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status=f"Failed: {http_exc.detail}"
        )
        db.add(training_run)
        db.commit()

        # Incrémenter les métriques Prometheus pour une réponse échouée
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status=str(http_exc.status_code)).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status=str(http_exc.status_code)).observe(time.time() - start_time)

        raise http_exc

    except Exception as e:
        # Enregistrer un run d'entraînement échoué avec une exception non HTTP
        training_run = TrainingRun(
            run_id=run_id,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status=f"Failed: {str(e)}"
        )
        db.add(training_run)
        db.commit()

        # Incrémenter les métriques Prometheus pour une réponse échouée
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status="500").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status="500").observe(time.time() - start_time)

        logger.error(f"Erreur lors de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=str(e))
