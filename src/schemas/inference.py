from pydantic import BaseModel
from typing import List, Dict

class InferenceRequest(BaseModel):
    file_path: str  # Chemin vers le fichier PDF/JPG/PNG

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: str
