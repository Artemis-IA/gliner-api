# src/schemas/inference.py
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class InferenceRequest(BaseModel):
    file_path: str = Field(..., example="path/to/file.pdf")

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime

    class Config:
        from_attributes = True
