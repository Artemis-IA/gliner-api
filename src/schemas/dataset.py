from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class DatasetCreate(BaseModel):
    name: str = Field(..., examples="Sample Dataset")
    data: List[Dict] = Field(..., examples=[{"text": "Example", "entities": ["Entity1", "Entity2"]}])

class DatasetUpdate(BaseModel):
    name: str = Field(..., examples="Updated Dataset")
    data: List[Dict] = Field(..., examples=[{"text": "Updated Example", "entities": ["Entity3"]}])

class DatasetResponse(BaseModel):
    id: int
    name: str
    data: List[Dict]
    created_at: datetime
