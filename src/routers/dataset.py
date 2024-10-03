from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpdate
from db.session import SessionLocal
from db.models import Dataset
from utils.gliner_utils import create_gliner_dataset
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
def create_dataset(request: DatasetCreate, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_create").inc()
    try:
        dataset_path = create_gliner_dataset(request.data, format="json-ner")
        dataset = Dataset(name=request.name, data=request.data)
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
