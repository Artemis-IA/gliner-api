# routers/dataset.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.schemas import NERDatasetResponse, NERDatasetSample
from services.dataset_creator import create_ner_dataset
from pathlib import Path
import shutil
import uuid
from typing import List

router = APIRouter(prefix="/dataset", tags=["Dataset"])

DATASET_DIR = Path("datasets")
DATASET_DIR.mkdir(exist_ok=True)
datasets_storage = {}  # Dictionnaire pour stocker les datasets en mémoire


@router.post("/", response_model=NERDatasetResponse)
async def create_dataset(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_location = DATASET_DIR / file.filename
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        file_paths.append(file_location)

    dataset = create_ner_dataset(file_paths)

    dataset_id = str(uuid.uuid4())
    datasets_storage[dataset_id] = dataset

    samples = [NERDatasetSample(text=item['text'], annotations=item['annotations']) for item in dataset]

    return NERDatasetResponse(dataset_id=dataset_id, samples=samples)


@router.get("/{dataset_id}", response_model=NERDatasetResponse)
async def get_dataset(dataset_id: str):
    dataset = datasets_storage.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset non trouvé.")

    samples = [NERDatasetSample(text=item['text'], annotations=item['annotations']) for item in dataset]

    return NERDatasetResponse(dataset_id=dataset_id, samples=samples)
