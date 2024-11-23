import psutil, GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from codecarbon import EmissionsTracker

# Metrics for Prometheus

REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")
MODEL_LOG_COUNT = Counter("model_log_count", "Nombre de modèles enregistrés dans MLflow")
NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Number of requests sent to Neo4j")
NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Number of failed Neo4j requests")
NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latency of Neo4j requests")

POSTGRES_QUERY_COUNT = Counter("postgres_query_count", "Number of successful PostgreSQL queries")
POSTGRES_QUERY_FAILURES = Counter("postgres_query_failures", "Number of failed PostgreSQL queries")
POSTGRES_QUERY_LATENCY = Histogram("postgres_query_latency_seconds", "Latency of PostgreSQL queries")

DOCUMENT_PROCESSING_SUCCESS = Counter("document_processing_success", "Number of successfully processed documents")
DOCUMENT_PROCESSING_FAILURES = Counter("document_processing_failures", "Number of failed document processing attempts")

emissions_tracker = EmissionsTracker(project_name="doc_processing", save_to_file=False, save_to_prometheus=True, prometheus_url="localhost:8002")

# Function to start the Prometheus metrics server
def start_metrics_server(port: int = 8002):
    """
    Start the Prometheus metrics server to expose application metrics.

    Args:
        port (int): The port to expose metrics on (default is 8002).
    """
    start_http_server(port)
    REQUEST_COUNT.inc()  # Increment the request count to indicate the server has started

def log_system_metrics():
    """
    Log et exposition des métriques système (CPU, RAM, GPU et émissions de CO₂).
    """
    try:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)

        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)  # Seulement la première GPU

        # CodeCarbon emissions
        global emissions_tracker
        if emissions_tracker:
            emissions_tracker.start()
            emissions = emissions_tracker.stop()
            if emissions is not None:
                CARBON_EMISSIONS.set(emissions)
                logger.info(f"Émissions collectées : {emissions:.6f} kgCO₂eq")
            else:
                logger.warning("Aucune donnée d'émissions collectée (None).")
    except Exception as e:
        logger.warning(f"Erreur lors de la collecte des métriques : {e}")