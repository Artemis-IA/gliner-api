# src/services/labelstudio_manager.py

from typing import List, Dict
from services.model_manager import ModelManager
import logging

class NERLabelStudioMLBackend:
    def __init__(self):
        self.model_manager = ModelManager()

    def start(self):
        """Initialise et charge le modèle."""
        logging.info("Initialisation du backend ML...")
        self.model_manager.load_model()
        logging.info("Backend ML prêt.")

    def stop(self):
        """Nettoie les ressources si nécessaire."""
        logging.info("Arrêt du backend ML...")
        # Implémentez la logique de nettoyage si nécessaire
        logging.info("Backend ML arrêté.")

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """Génère des prédictions pour les tâches fournies."""
        texts = [task['data']['text'] for task in tasks]
        predictions = self.model_manager.predict(
            texts=texts,
            labels=None,  # Spécifiez les étiquettes si nécessaire
            flat_ner=True,
            threshold=0.3,
            multi_label=False,
            batch_size=12
        )

        results = []
        for task, prediction in zip(tasks, predictions):
            result = []
            for entity in prediction:
                result.append({
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": entity['start'],
                        "end": entity['end'],
                        "labels": [entity['label']]
                    }
                })
            results.append({"result": result})

        return results

    def fit(self, completions: List[Dict], **kwargs):
        """Entraîne le modèle avec les annotations fournies."""
        annotated_data = []
        for completion in completions:
            text = completion['data']['text']
            annotations = completion['annotations']
            entities = []
            for ann in annotations:
                for label in ann['value']['labels']:
                    entities.append({
                        'start': ann['value']['start'],
                        'end': ann['value']['end'],
                        'label': label
                    })
            annotated_data.append({
                'text': text,
                'entities': entities
            })
        
        self.model_manager.train_model(train_data=annotated_data)
