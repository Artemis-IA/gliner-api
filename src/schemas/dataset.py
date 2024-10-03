from pydantic import BaseModel
from typing import List, Dict

class DatasetCreate(BaseModel):
    name: str
    data: List[Dict]  # Liste d'entités NER

class DatasetUpdate(BaseModel):
    name: str
    data: List[Dict]

class DatasetResponse(BaseModel):
    id: int
    name: str
    data: List[Dict]
    created_at: str
