# routers/ner.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from models.ner_model import NERModel
from schemas.schemas import NERInferenceResponse, ModelListResponse, Entity
from utils.file_utils import extract_text_from_pdf, extract_text_from_image
from services.model_manager import ModelManager
from pathlib import Path
import shutil
from typing import List

router = APIRouter(prefix="/ner", tags=["NER"])

model_manager = ModelManager()
available_models = model_manager.get_available_models()


@router.get("/models", response_model=ModelListResponse)
async def get_models():
    """Récupère la liste des modèles GLiNER disponibles."""
    return ModelListResponse(models=list(available_models.keys()))


@router.post("/predict", response_model=NERInferenceResponse)
async def perform_ner(
    file: UploadFile = File(...),
    model_name: str = "GLiNER-S",
    labels: List[str] = None,
    threshold: float = 0.5,
):
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Modèle non disponible.")

    ner_model = NERModel(name=model_name)
    ner_model.load()

    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_location = UPLOAD_DIR / file.filename

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_location)
    elif file.content_type in ["image/png", "image/jpeg"]:
        text = extract_text_from_image(file_location)
    else:
        raise HTTPException(status_code=400, detail="Type de fichier non supporté")

    texts = [text]

    # Utiliser des labels par défaut si aucun n'est fourni
    if labels is None:
        labels = ["Person", "Organization", "Location", "Date", "Time", "Money", "Percent"]

    # Effectuer la prédiction
    predictions = ner_model.batch_predict(
        targets=texts,
        labels=labels,
        threshold=threshold,
    )

    # Traiter les prédictions pour correspondre au schéma NERInferenceResponse
    entities = []
    for entity_list in predictions:
        for entity in entity_list:
            entities.append(Entity(
                text=entity['text'],
                label=entity['label'],
                start_char=entity['start'],
                end_char=entity['end']
            ))

    return NERInferenceResponse(filename=file.filename, entities=entities)
