# src/main.py

from fastapi import FastAPI
from routers import auth, dataset, inference, train, ml_backend
from utils.mlflow_manager import MLflowManager
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from db.init_db import init_db
import uvicorn
import logging

app = FastAPI(
    title="GLiNER CRUD API",
    description="API CRUD pour le projet GLiNER avec FastAPI, MLflow et Prometheus",
    version="1.0.0"
)

mlflow_manager = MLflowManager()

@app.on_event("startup")
def on_startup():
    logging.info("Démarrage de l'application...")
    # init_db()
    # mlflow_manager.setup_mlflow()
    # mlflow_manager.run_migrations()
    # ml_backend.start()


@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).observe(process_time)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()

    return response

app.include_router(auth.router, prefix="/auth")
app.include_router(dataset.router, prefix="/dataset")
app.include_router(inference.router, prefix="/inference")
app.include_router(train.router, prefix="/train")
app.include_router(ml_backend.router, prefix="/ml")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
