# main.py

from fastapi import FastAPI
from routers import inference, dataset, train
from utils.mlflow_setup import setup_mlflow
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="GLiNER CRUD API",
    description="API CRUD pour le projet GLiNER avec FastAPI, MLflow et Prometheus",
    version="1.0.0"
)

# Configurer MLflow
setup_mlflow()

# Inclure les routers
app.include_router(inference.router)
app.include_router(dataset.router)
app.include_router(train.router)

# Ajouter middleware CORS si nécessaire
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modifier selon les besoins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajouter l'endpoint Prometheus
app.mount("/metrics", make_asgi_app())

# Point de terminaison racine
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API GLiNER CRUD"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
