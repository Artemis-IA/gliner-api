# utils/metrics.py

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    'request_count', 'Total number of requests', ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of HTTP requests')

def setup_metrics(app: FastAPI):

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
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
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
