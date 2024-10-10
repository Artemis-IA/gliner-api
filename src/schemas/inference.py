# src/schemas/inference.py
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from fastapi import UploadFile
class InferenceRequest(BaseModel):
    files: List[UploadFile] = Field(..., description="Fichiers PDF à traiter")
    labels: List[str] = Field(..., description="Liste des labels pour l'inférence")

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime

    class Config:
        from_attributes = True
