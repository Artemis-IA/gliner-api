# utils/metrics.py
import os, psutil, GPUtil
import torch
from loguru import logger 
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from codecarbon import EmissionsTracker

# Metrics for Prometheus

REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")

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
    logger.info(f"Prometheus metrics server started on port {port}.")




def log_system_metrics():
    """
    Log system metrics such as CPU, memory, GPU usage, and CO2 emissions.
    """
    try:
        # Log CPU and memory usage
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)

        # Log GPU memory usage
        gpus = GPUtil.getGPUs()
        if gpus:
            GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)

        # Log CO2 emissions
        emissions = emissions_tracker.stop()
        if emissions is not None:
            CARBON_EMISSIONS.set(emissions)
            logger.info(f"CO2 emissions logged: {emissions:.6f} kgCO₂eq")
        else:
            logger.warning("No emissions data available.")
    except Exception as e:
        logger.warning(f"Error logging system metrics: {e}")


@staticmethod
def get_system_metrics() -> dict:
    """
    Retrieve system metrics (CPU, RAM, GPU) and hardware setup information.
    """
    # Basic metrics
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(),
        "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024),  # Convert to MB
    }

    # GPU metrics
    gpus = GPUtil.getGPUs()
    if gpus:
        metrics["gpu_memory_usage_mb"] = gpus[0].memoryUsed  # First GPU only
    else:
        metrics["gpu_memory_usage_mb"] = None

    # Check CUDA or CPU usage
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        metrics["cuda"] = True
        metrics["gpu_name"] = gpu_name
        logger.info(f"CUDA is available. Using GPU: {gpu_name}")
    else:
        metrics["cuda"] = False
        logger.info("CUDA is not available. Using CPU for processing.")

    logger.info(f"System metrics: {metrics}")
    return metrics

def _remove_codecarbon_lock():
    """
    Remove CodeCarbon lock file at startup to avoid tracker errors.
    """
    lock_file = "/tmp/.codecarbon.lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            logger.info("CodeCarbon lock file removed.")
        except Exception as e:
            logger.warning(f"Error removing CodeCarbon lock file: {e}")