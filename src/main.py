# main.py

from fastapi import FastAPI
from routers import ner, train, auth, dataset
from utils.metrics import setup_metrics

app = FastAPI(title="NER API", version="2.0")

# Inclure les routeurs
app.include_router(auth.router)
app.include_router(ner.router)
app.include_router(train.router)
app.include_router(dataset.router)


# Configurer les métriques Prometheus
setup_metrics(app)
