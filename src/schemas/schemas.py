# schemas/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class NERInferenceResponse(BaseModel):
    filename: str
    entities: List[Entity]


class NERDatasetSample(BaseModel):
    text: str
    annotations: List[Entity]


class NERDatasetResponse(BaseModel):
    dataset_id: str
    samples: List[NERDatasetSample]


class TrainRequest(BaseModel):
    model_name: str
    labels: List[str]
    dataset_id: str
    epochs: int = 3
    batch_size: int = 16


class TrainResponse(BaseModel):
    status: str
    model_path: Optional[str]


class ModelListResponse(BaseModel):
    models: List[str]
