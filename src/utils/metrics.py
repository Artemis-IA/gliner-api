# src/utils/metrics.py

from prometheus_client import Counter, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from fastapi import Response
# Utilisation d'un registre spécifique
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    'request_count', 'Total number of requests', ['method', 'endpoint', 'http_status'], registry=registry
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Latency of HTTP requests', 
    ['method', 'endpoint', 'http_status'],
    registry=registry
)

def setup_metrics(app):

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        REQUEST_LATENCY.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status=response.status_code
        ).inc()

        return response

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry=registry), media_type=CONTENT_TYPE_LATEST)
