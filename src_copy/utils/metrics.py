# src/utils/metrics.py
import os
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from codecarbon import EmissionsTracker


class MetricBase:
    """Base class for Prometheus metrics."""
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry


class RequestMetrics(MetricBase):
    """Prometheus metrics related to HTTP requests."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.request_count = Counter(
            "app_request_count", 
            "Nombre total de requêtes",
            ["method", "endpoint", "http_status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "app_request_latency_seconds", 
            "Temps de latence des requêtes",
            ["method", "endpoint", "http_status"],
            registry=self.registry,
        )
        self.process_time = Histogram(
            "app_process_time_seconds", 
            "Temps de traitement des requêtes",
            registry=self.registry,
        )


class SystemMetrics(MetricBase):
    """Prometheus metrics for system usage."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.gpu_memory_usage = Gauge(
            "gpu_memory_usage_bytes", 
            "Utilisation mémoire GPU", 
            registry=self.registry,
        )
        self.cpu_usage = Gauge(
            "cpu_usage_percent", 
            "Utilisation CPU", 
            registry=self.registry,
        )
        self.memory_usage = Gauge(
            "memory_usage_bytes", 
            "Utilisation mémoire RAM", 
            registry=self.registry,
        )


class DatabaseMetrics(MetricBase):
    """Prometheus metrics for database usage."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.postgres_query_count = Counter(
            "postgres_query_count", 
            "Nombre de requêtes PostgreSQL réussies", 
            registry=self.registry,
        )
        self.postgres_query_failures = Counter(
            "postgres_query_failures", 
            "Nombre de requêtes PostgreSQL échouées", 
            registry=self.registry,
        )
        self.postgres_query_latency = Histogram(
            "postgres_query_latency_seconds", 
            "Latence des requêtes PostgreSQL", 
            registry=self.registry,
        )


class Neo4jMetrics(MetricBase):
    """Prometheus metrics for Neo4j usage."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.neo4j_request_count = Counter(
            "neo4j_request_count", 
            "Nombre de requêtes envoyées à Neo4j", 
            registry=self.registry,
        )
        self.neo4j_request_failures = Counter(
            "neo4j_request_failures", 
            "Nombre de requêtes Neo4j échouées", 
            registry=self.registry,
        )
        self.neo4j_request_latency = Histogram(
            "neo4j_request_latency_seconds", 
            "Latence des requêtes Neo4j", 
            registry=self.registry,
        )


class DocumentProcessingMetrics(MetricBase):
    """Prometheus metrics for document processing."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.success = Counter(
            "document_processing_success", 
            "Nombre de documents traités avec succès", 
            registry=self.registry,
        )
        self.failures = Counter(
            "document_processing_failures", 
            "Nombre d'échecs de traitement de documents", 
            registry=self.registry,
        )


class CarbonMetrics(MetricBase):
    """Prometheus metrics for carbon emissions."""
    def __init__(self, registry: CollectorRegistry):
        super().__init__(registry)
        self.carbon_emissions = Gauge(
            "carbon_emissions_grams", 
            "Émissions CO2 estimées", 
            registry=self.registry,
        )
        self.tracker = EmissionsTracker(
            project_name="doc_processing",
            save_to_file=False,
            save_to_prometheus=True,
        )

    def start_emissions_tracker(self):
        """Start the carbon emissions tracker."""
        try:
            self.tracker.start()
            logger.info("Emissions tracker started.")
        except Exception as e:
            logger.warning(f"Error starting emissions tracker: {e}")

    def stop_emissions_tracker(self):
        """Stop the carbon emissions tracker and log emissions."""
        try:
            emissions = self.tracker.stop()
            if emissions is not None:
                self.carbon_emissions.set(emissions)
                logger.info(f"Carbon emissions recorded: {emissions:.6f} grams")
        except Exception as e:
            logger.warning(f"Error stopping emissions tracker: {e}")


class MetricsManager:
    """Manager to handle all metrics."""
    def __init__(self):
        self.registry = CollectorRegistry()
        self.request_metrics = RequestMetrics(self.registry)
        self.system_metrics = SystemMetrics(self.registry)
        self.database_metrics = DatabaseMetrics(self.registry)
        self.neo4j_metrics = Neo4jMetrics(self.registry)
        self.document_processing_metrics = DocumentProcessingMetrics(self.registry)
        self.carbon_metrics = CarbonMetrics(self.registry)

    def log_system_metrics(self):
        """Log CPU, memory, and GPU metrics."""
        try:
            self.system_metrics.cpu_usage.set(psutil.cpu_percent())
            self.system_metrics.memory_usage.set(psutil.virtual_memory().used)
            gpus = GPUtil.getGPUs()
            if gpus:
                self.system_metrics.gpu_memory_usage.set(gpus[0].memoryUsed)
        except Exception as e:
            logger.warning(f"Error logging system metrics: {e}")

    def expose_metrics(self, app):
        """Expose metrics through the FastAPI app."""
        @app.middleware("http")
        async def metrics_middleware(request, call_next):
            import time
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time

            self.request_metrics.request_latency.labels(
                method=request.method,
                endpoint=request.url.path,
                http_status=response.status_code,
            ).observe(process_time)
            self.request_metrics.request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                http_status=response.status_code,
            ).inc()

            self.log_system_metrics()
            return response

        # Configure the Prometheus Instrumentator
        Instrumentator(
            excluded_handlers=["^/health", "^/docs"],  # Correct usage
            should_group_status_codes=True,
            should_ignore_untemplated=True,
        ).instrument(app).expose(app)


        @app.get("/metrics")
        async def metrics():
            return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)

metrics_manager = MetricsManager()