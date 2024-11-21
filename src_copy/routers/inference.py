# src/routers/inference.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, Form
from sqlalchemy.orm import Session
from pathlib import Path
from schemas.inference import InferenceRequest, InferenceResponse
from schemas.models_dict import ModelName
from db.session import SessionLocal
from db.models import Inference
from utils.file_utils import FileProcessor
from models.ner_model import NERModel, MODELS
from loguru import logger
import time
from typing import List


# Create an instance of the NERModel
ner_model_instance = NERModel(name="GLiNER-S")

router = APIRouter(
    prefix="/predict",
    tags=["Inference"]
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def process_file_and_inference(file_content: bytes, file_name: str, labels: str, threshold: float, db: Session, predictions: List):
    file_path = Path(f"/tmp/{file_name}")
    logger.info(f"Processing file: {file_name}")

    # Save file temporarily
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # Extract text from the file
    text = FileProcessor.process_file(file_path)
    if text is None:
        logger.error(f"Failed to extract text from {file_name}")
        raise HTTPException(status_code=400, detail="Failed to extract text from file")

    # Predict entities
    raw_entities = ner_model_instance.batch_predict([text], labels.split(","))
    logger.info(f"Predicted entities: {raw_entities}")

    # Flatten entities and format them properly
    entities = []
    for entity_group in raw_entities:
        for entity in entity_group:
            entities.append({
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": entity["label"],
                "score": entity["score"]
            })

    # Save inference in the database
    inference = Inference(file_path=file_name, entities=entities)
    db.add(inference)
    db.commit()
    db.refresh(inference)
    logger.info(f"Inference saved to DB with ID: {inference.id}")

    # Add to predictions list
    predictions.append(InferenceResponse(
        id=inference.id,
        file_path=inference.file_path,
        entities=inference.entities,
        created_at=inference.created_at.isoformat()
    ))

@router.post("/", response_model=List[InferenceResponse], status_code=202)
async def predict_endpoint(
    inference_request: InferenceRequest = Depends(InferenceRequest.as_form),
    selected_model: ModelName = Form(..., description="Sélectionnez le modèle NER"),
    files: List[UploadFile] = File(..., description="Fichiers à traiter"),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    predictions = []

    if selected_model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle {selected_model} non valide. Choisissez parmi: {', '.join(MODELS.keys())}")

    try:
        # Charger le modèle sélectionné
        logger.info(f"Chargement du modèle {selected_model}")
        ner_model_instance = NERModel(name=selected_model)
        ner_model_instance.load()  # Charger le modèle sélectionné

        # Process each file sequentially and synchronously
        for file in files:
            logger.info(f"Received file: {file.filename}")
            file_content = await file.read()  # Read file content
            process_file_and_inference(
                file_content, 
                file.filename, 
                ",".join(inference_request.labels), 
                inference_request.threshold, 
                db, 
                predictions
            )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        logger.info(f"Inference completed in {latency:.2f} seconds")

    logger.info(f"Returning predictions: {predictions}")
    return predictions
