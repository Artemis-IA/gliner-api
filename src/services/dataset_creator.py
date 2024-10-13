# src/services/dataset_creator.py
from typing import List, Dict, Optional
from fastapi import UploadFile
from pathlib import Path
from utils.file_utils import FileProcessor
from minio import Minio
from sqlalchemy.orm import Session
from db.models import Dataset
from core.config import settings
import os

# Setup MinIO client
minio_client = Minio(
    "localhost:9000",  # settings.minio_api_url,
    access_key=settings.minio_root_user,
    secret_key=settings.minio_root_password,
    secure=False  # Set to True if using HTTPS
)

bucket_name = "datasets"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

async def upload_to_minio(file_path: Path, name: str, file_type: str) -> str:
    """Upload a file to MinIO and return its URL."""
    object_name = f"{name}/{file_type}/{file_path.name}"
    minio_client.fput_object(bucket_name, object_name, str(file_path))
    file_url = f"http://{settings.minio_api_url}/{bucket_name}/{object_name}"
    return file_url

async def create_ner_dataset(
    files: List[UploadFile],
    output_format: str,
    labels: Optional[List[str]],
    name: str,
    db: Session
) -> List[Dict]:
    """Create a dataset for NER tasks from the provided PDF files and store in MinIO."""
    dataset = []
    output_dir = Path(f"/tmp/extracted_data/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = output_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the file: extract text and bounding boxes, convert PDF to images
        file_data = FileProcessor.process_file(file_path, output_dir)

        # Upload images to MinIO and get their URLs
        image_urls = []
        for image_path in file_data["images"]:
            image_url = await upload_to_minio(image_path, name, "images")
            image_urls.append(image_url)

        # Upload text file to MinIO (optional, if you want the text separately)
        text_file_path = output_dir / f"{file.filename}.txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(file_data["text"])
        text_url = await upload_to_minio(text_file_path, name, "text")

        # Prepare the dataset entry for Label Studio
        dataset.append({
            'task_id': len(dataset) + 1,  # Unique task ID
            'image_urls': image_urls,
            'text_url': text_url,
            'boxes': file_data["boxes"],  # Bounding boxes of text
            'file_name': file.filename,
            'annotations': []  # No annotations initially
        })

    # Save the dataset metadata in PostgreSQL
    dataset_entry = Dataset(name=name, data=dataset)
    db.add(dataset_entry)
    db.commit()
    db.refresh(dataset_entry)

    # Generate tasks in Label Studio format
    label_studio_tasks = format_for_label_studio(dataset)
    return label_studio_tasks

def format_for_label_studio(dataset: List[Dict]) -> List[Dict]:
    """Convert the dataset into a format compatible with Label Studio."""
    label_studio_dataset = []
    for data in dataset:
        # Create a task for each image in the PDF
        for image_url, box in zip(data["image_urls"], data["boxes"]):
            task = {
                "data": {
                    "image": image_url,  # Image URL for Label Studio to display
                    "text": data["text_url"]  # Text associated with the image (in MinIO)
                },
                "meta": {
                    "source": data["file_name"],
                    "task_id": data["task_id"]
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "from_name": "bbox",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "value": {
                                    "x": box['bbox'][0],
                                    "y": box['bbox'][1],
                                    "width": box['bbox'][2] - box['bbox'][0],
                                    "height": box['bbox'][3] - box['bbox'][1]
                                }
                            }
                        ]
                    }
                ]
            }
            label_studio_dataset.append(task)
    return label_studio_dataset
