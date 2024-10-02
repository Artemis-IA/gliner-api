# services/model_manager.py

from typing import Dict
from loguru import logger
from models.ner_model import MODELS


class ModelManager:
    """Gère la récupération des modèles GLiNER disponibles."""

    def __init__(self):
        pass  # Pas besoin d'initialiser HfApi

    def get_available_models(self) -> Dict[str, str]:
        """Récupère les modèles GLiNER disponibles.

        :return: Dictionnaire des modèles disponibles.
        """
        model_dict = MODELS
        logger.info(f"{len(model_dict)} modèles GLiNER disponibles récupérés.")
        return model_dict
