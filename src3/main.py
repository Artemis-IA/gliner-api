import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import start_http_server
from routers import documents, entities, relationships, search, graph
from utils.metrics import REQUEST_COUNT, PROCESS_TIME, log_system_metrics
from config import settings

app = FastAPI(title="Document Processing and Graph API", version="2.0.0")

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(documents.router)
app.include_router(entities.router)
app.include_router(relationships.router)
app.include_router(search.router)
app.include_router(graph.router)

# Prometheus Metrics
start_http_server(8002)

@app.middleware("http")
async def custom_metrics_middleware(request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    response = await call_next(request)
    latency = time.time() - start_time
    PROCESS_TIME.observe(latency)
    log_system_metrics()  # Log system metrics
    return response

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
