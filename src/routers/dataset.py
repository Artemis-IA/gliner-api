# src/routers/dataset.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from services.dataset_creator import create_ner_dataset
from schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpdate
from db.session import SessionLocal
from db.models import Dataset
from sqlalchemy.orm import Session
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
import time

router = APIRouter(
    prefix="/datasets",
    tags=["Dataset Creator"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    name: str,
    format: str = "json-ner",
    files: List[UploadFile] = File(...)
, db: Session = Depends(get_db)):
    """Endpoint pour créer un dataset à partir de fichiers uploadés."""
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_create").inc()
    try:
        # Créer le dataset
        dataset_data = await create_ner_dataset(files, format=format)
        # Enregistrer dans la base de données
        dataset = Dataset(name=name, data=dataset_data)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="dataset_create").observe(latency)

@router.get("/{dataset_id}", response_model=DatasetResponse)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_read").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="dataset_read").observe(latency)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        data=dataset.data,
        created_at=dataset.created_at.isoformat()
    )

@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(dataset_id: int, request: DatasetUpdate, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_update").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        dataset.name = request.name
        dataset.data = request.data
        db.commit()
        db.refresh(dataset)
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="dataset_update").observe(latency)

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_delete").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        db.delete(dataset)
        db.commit()
        return {"detail": "Dataset deleted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="dataset_delete").observe(latency)
