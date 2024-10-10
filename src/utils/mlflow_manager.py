# src/utils/mlflow_manager.py

import mlflow
import subprocess
from core.config import settings
from mlflow.exceptions import MlflowException
import logging

class MLflowManager:
    def __init__(self):
        self.tracking_uri = settings.mlflow_tracking_uri
        self.backend_store_uri = settings.mlflow_backend_store_uri
        self.artifact_root = settings.mlflow_artifact_root

    def setup_mlflow(self):
        # Définir l'URI de suivi MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            # Essayer de définir un experiment
            mlflow.set_experiment("GLiNER_Experiments")
            logging.info("MLflow experiment 'GLiNER_Experiments' is set.")
        except MlflowException as e:
            logging.warning(f"MLflow Exception: {e}")
            logging.info("Il semble que les tables MLflow soient manquantes. Tentative d'initialisation de la base de données.")
            self.run_migrations()
            # Réessayer de définir l'experiment après la création des tables
            mlflow.set_experiment("GLiNER_Experiments")
            logging.info("MLflow experiment 'GLiNER_Experiments' is set after migrations.")

    def run_migrations(self):
        try:
            # Exécuter la commande de migration MLflow
            logging.info("Exécution des migrations MLflow...")
            subprocess.run(
                [
                    "mlflow", "db", "upgrade",
                    self.backend_store_uri
                ],
                check=True
            )
            logging.info("Tables de la base de données MLflow créées avec succès.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Erreur lors de la création des tables MLflow : {e}")
