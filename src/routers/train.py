# routers/train.py

from fastapi import APIRouter, HTTPException
from models.ner_model import NERModel
from schemas.schemas import TrainRequest, TrainResponse
from services.model_manager import ModelManager
from routers.dataset import datasets_storage
from pathlib import Path

router = APIRouter(prefix="/train", tags=["Train"])

model_manager = ModelManager()
available_models = model_manager.get_available_models()


@router.post("/", response_model=TrainResponse)
async def train_model(train_request: TrainRequest):
    model_name = train_request.model_name
    labels = train_request.labels
    dataset_id = train_request.dataset_id
    epochs = train_request.epochs
    batch_size = train_request.batch_size

    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Modèle non disponible.")

    dataset = datasets_storage.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset non trouvé.")

    train_texts = [item['text'] for item in dataset]
    train_annotations = [item['annotations'] for item in dataset]

    # Préparer les données d'entraînement dans le format attendu par GLiNER
    train_data = []
    for text, annotations in zip(train_texts, train_annotations):
        train_data.append({
            'text': text,
            'entities': annotations  # Doit être au format attendu par GLiNER
        })

    ner_model = NERModel(name=model_name)
    ner_model.load()

    # Ajuster la configuration d'entraînement
    ner_model.train_config.num_steps = epochs * len(train_data) // batch_size
    ner_model.train_config.train_batch_size = batch_size

    # Entraîner le modèle
    ner_model.train(
        train_data=train_data,
        eval_data=None,  # Ajouter des données d'évaluation si disponible
    )

    # Sauvegarder le modèle entraîné
    model_output_dir = f"trained_models/{model_name.replace('/', '_')}_{dataset_id}"
    ner_model.save(file_name=model_output_dir)

    return TrainResponse(status="training_completed", model_path=model_output_dir)
