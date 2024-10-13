# src/routers/dataset.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from schemas.dataset import DatasetUpdate
from typing import List
from fastapi.responses import JSONResponse
from services.dataset_creator import create_ner_dataset
from schemas.dataset import DatasetResponse
from db.session import SessionLocal
from db.models import Dataset
from sqlalchemy.orm import Session
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
import time

router = APIRouter(
    prefix="/datasets",
    tags=["Dataset Creator"]
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    files: List[UploadFile] = File(...),
    name: str = Form(None),
    labels: str = Form("PERSON,ORG,GPE,DATE"),
    output_format: str = Form("json"),
    db: Session = Depends(get_db)
):
    """
    Endpoint to create an NER dataset from uploaded files, with optional name and format.
    If no name is provided, the name will default to the first PDF's filename + dataset ID.
    """
    start_time = time.time()

    try:
        # Convert labels string to list
        labels_list = [label.strip() for label in labels.split(',')] if labels else None

        # Create the dataset
        dataset_data = await create_ner_dataset(files, output_format=output_format, labels=labels_list, name=name, db=db)

        # Determine default name if not provided
        if not name:
            pdf_files = [file for file in files if file.filename.endswith('.pdf')]
            if pdf_files:
                default_name = pdf_files[0].filename.rsplit('.', 1)[0]  # Filename without extension
            else:
                default_name = "dataset"

            # Get next auto-increment ID from the database for naming
            next_id = db.execute("SELECT nextval('datasets_id_seq')").scalar()
            name = f"{default_name}_{next_id}"

        # Save dataset to the database
        dataset = Dataset(name=name, data=dataset_data)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        # Increment request count with the correct labels
        REQUEST_COUNT.labels(method="POST", endpoint="/datasets/", http_status="200").inc()

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )

    except Exception as e:
        db.rollback()
        REQUEST_COUNT.labels(method="POST", endpoint="/datasets/", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method="POST", endpoint="/datasets/", http_status="200").observe(latency)
        print(f"Dataset creation completed in {latency:.2f} seconds")

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
