# src/routers/ml_backend.py

from fastapi import APIRouter, Request, HTTPException
from typing import List, Dict
from services.labelstudio_manager import NERLabelStudioMLBackend

router = APIRouter()

# Instanciation du backend ML
ml_backend = NERLabelStudioMLBackend()

router = APIRouter(
    prefix="/ml_backend",
    tags=["ML Backend"]
)

@router.get("/health")
async def health():
    """Endpoint de santé pour vérifier si le backend ML est opérationnel."""
    try:
        # Vous pouvez ajouter des vérifications supplémentaires si nécessaire
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(request: Request):
    """Endpoint pour générer des prédictions."""
    try:
        tasks = await request.json()
        results = ml_backend.predict(tasks)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fit")
async def fit(request: Request):
    """Endpoint pour entraîner le modèle avec des annotations."""
    try:
        completions = await request.json()
        ml_backend.fit(completions)
        return {"status": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
