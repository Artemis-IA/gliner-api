import time
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from routers import documents, entities, relationships, search, graph
from utils.metrics import start_metrics_server, REQUEST_COUNT, PROCESS_TIME, log_system_metrics, _remove_codecarbon_lock, get_system_metrics
from config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Processing and Graph API",
        version="2.0.0",
        description="A platform for document processing, graph indexing and NER/Relation extraction."
    )

    # Middleware for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware
    Instrumentator(
        excluded_handlers=["^/metrics", "^/docs"],
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app)

    # Include Routers
    app.include_router(documents.router)
    app.include_router(entities.router)
    app.include_router(relationships.router)
    app.include_router(search.router)
    app.include_router(graph.router)

    @app.middleware("http")
    async def custom_metrics_middleware(request, call_next):
        """Custom middleware to track request metrics."""
        start_time = time.time()
        REQUEST_COUNT.inc()
        response = await call_next(request)
        latency = time.time() - start_time
        PROCESS_TIME.observe(latency)
        log_system_metrics()  # Log system metrics
        return response

    @app.get("/metrics")
    async def metrics():
        """Expose Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    # _remove_codecarbon_lock()
    logger.info("Application starting...")
    start_metrics_server(port=8002)

    # Log system metrics on startup
    system_metrics = get_system_metrics()
    device_type = "GPU" if system_metrics["cuda"] else "CPU"


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
