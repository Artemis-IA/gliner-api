# src/services/model_manager.py
from models.ner_model import NERModel
from typing import List
from threading import Lock
from typing import Optional
from loguru import logger

class ModelManager:
    """Gestionnaire du modèle NER."""

    def __init__(self):
        self.model: Optional[NERModel] = None
        self.lock = Lock()

    def load_model(self, name: str = "GLiNER-S") -> None:
        with self.lock:
            if self.model is None:
                logger.info("Chargement du modèle NER...")
                self.model = NERModel(name=name)
                self.model.load()
                logger.info("Modèle NER chargé.")
            else:
                logger.info("Le modèle est déjà chargé.")

    def predict(
        self,
        texts: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[dict]]:
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        return self.model.batch_predict(
            targets=texts,
            labels=labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
        )

    def train_model(
        self,
        train_data: List[dict],
        eval_data: dict = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        self.model.train(train_data, eval_data)
