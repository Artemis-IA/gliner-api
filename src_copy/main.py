from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from routers import auth, dataset, inference, train, ml_backend, mlflow
from utils.metrics import metrics_manager
from db.init_db import init_db
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)

class DocGraphAPI:
    def __init__(self):
        self._remove_codecarbon_lock()  # Nettoyer les verrous au démarrage
        self.app = FastAPI(
            title="DocGraph API & NERTrainer",
            description=(
                "DocGraph API & NERTrainer est une solution complète pour la conversion, "
                "l'analyse sémantique, l'indexation, et la gestion de graphes de connaissances, "
                "avec des fonctionnalités avancées pour le fine-tuning de modèles NER, la gestion "
                "des artefacts avec MinIO, et le suivi des performances via MLflow et Prometheus."
            ),
            version="1.0.0",
        )
        self._setup_middlewares()
        self._include_routes()
        self._add_prometheus_metrics()

    def _remove_codecarbon_lock(self):
        """Supprime le fichier de verrouillage de CodeCarbon au démarrage."""
        lock_file = "/tmp/.codecarbon.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logging.info("CodeCarbon lock file removed.")
            except Exception as e:
                logging.warning(f"Error removing CodeCarbon lock file: {e}")

    def _setup_middlewares(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Restreindre en production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _include_routes(self):
        self.app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
        self.app.include_router(dataset.router, prefix="/dataset", tags=["Dataset Management"])
        self.app.include_router(inference.router, prefix="/inference", tags=["Inference"])
        self.app.include_router(train.router, prefix="/train", tags=["Model Training"])
        self.app.include_router(mlflow.router, prefix="/mlflow", tags=["Model Logging & Management"])
        self.app.include_router(ml_backend.router, prefix="/ml", tags=["ML Backend"])

    def _add_prometheus_metrics(self):
        metrics_manager.expose_metrics(self.app)  # Expose Prometheus metrics

    async def _on_startup(self):
        logging.info("Démarrage de l'application...")
        init_db()
        metrics_manager.carbon_metrics.start_emissions_tracker()
        logging.info("Initialisation terminée.")

    async def _on_shutdown(self):
        metrics_manager.carbon_metrics.stop_emissions_tracker()
        logging.info("Arrêt de l'application.")

    def add_lifecycle_events(self):
        self.app.on_event("startup")(self._on_startup)
        self.app.on_event("shutdown")(self._on_shutdown)


docgraph_api = DocGraphAPI()
docgraph_api.add_lifecycle_events()

if __name__ == "__main__":
    uvicorn.run(
        "main:docgraph_api",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )
