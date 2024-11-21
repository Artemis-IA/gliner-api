from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from schemas.dataset import DatasetUpdate, DatasetResponse
from typing import List
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import Dataset
from services.dataset_creator import create_ner_dataset
from utils.metrics import metrics_manager
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

        # Increment metrics
        metrics_manager.request_count.labels(method="POST", endpoint="/datasets/", http_status="200").inc()
        latency = time.time() - start_time
        metrics_manager.request_latency.labels(method="POST", endpoint="/datasets/", http_status="200").observe(latency)

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )

    except Exception as e:
        db.rollback()
        metrics_manager.request_count.labels(method="POST", endpoint="/datasets/", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}", response_model=DatasetResponse)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to read a dataset by ID.
    """
    start_time = time.time()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            metrics_manager.request_count.labels(method="GET", endpoint="/datasets/{dataset_id}", http_status="404").inc()
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Metrics
        metrics_manager.request_count.labels(method="GET", endpoint="/datasets/{dataset_id}", http_status="200").inc()
        latency = time.time() - start_time
        metrics_manager.request_latency.labels(method="GET", endpoint="/datasets/{dataset_id}", http_status="200").observe(latency)

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )
    except Exception as e:
        metrics_manager.request_count.labels(method="GET", endpoint="/datasets/{dataset_id}", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(dataset_id: int, request: DatasetUpdate, db: Session = Depends(get_db)):
    """
    Endpoint to update a dataset by ID.
    """
    start_time = time.time()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            metrics_manager.request_count.labels(method="PUT", endpoint="/datasets/{dataset_id}", http_status="404").inc()
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset.name = request.name
        dataset.data = request.data
        db.commit()
        db.refresh(dataset)

        # Metrics
        metrics_manager.request_count.labels(method="PUT", endpoint="/datasets/{dataset_id}", http_status="200").inc()
        latency = time.time() - start_time
        metrics_manager.request_latency.labels(method="PUT", endpoint="/datasets/{dataset_id}", http_status="200").observe(latency)

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )
    except Exception as e:
        db.rollback()
        metrics_manager.request_count.labels(method="PUT", endpoint="/datasets/{dataset_id}", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to delete a dataset by ID.
    """
    start_time = time.time()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            metrics_manager.request_count.labels(method="DELETE", endpoint="/datasets/{dataset_id}", http_status="404").inc()
            raise HTTPException(status_code=404, detail="Dataset not found")

        db.delete(dataset)
        db.commit()

        # Metrics
        metrics_manager.request_count.labels(method="DELETE", endpoint="/datasets/{dataset_id}", http_status="200").inc()
        latency = time.time() - start_time
        metrics_manager.request_latency.labels(method="DELETE", endpoint="/datasets/{dataset_id}", http_status="200").observe(latency)

        return {"detail": "Dataset deleted"}
    except Exception as e:
        db.rollback()
        metrics_manager.request_count.labels(method="DELETE", endpoint="/datasets/{dataset_id}", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
