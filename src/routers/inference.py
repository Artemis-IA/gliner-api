from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from schemas.inference import InferenceRequest, InferenceResponse
from db.session import SessionLocal
from db.models import Inference
from utils.gliner_utils import run_gliner_inference
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
import time

router = APIRouter(
    prefix="/inference",
    tags=["Inference"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=InferenceResponse)
def create_inference(request: InferenceRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="inference_create").inc()
    try:
        entities = run_gliner_inference(request.file_path)
        inference = Inference(file_path=request.file_path, entities=entities)
        db.add(inference)
        db.commit()
        db.refresh(inference)
        return InferenceResponse(
            id=inference.id,
            file_path=inference.file_path,
            entities=inference.entities,
            created_at=inference.created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="inference_create").observe(latency)
