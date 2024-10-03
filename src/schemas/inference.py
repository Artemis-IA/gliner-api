from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class InferenceRequest(BaseModel):
    file_path: str = Field(..., examples="path/to/file.pdf")

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime
