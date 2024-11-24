
# config.py
#-----
import os
from pathlib import Path
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # Application settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Proxy settings
    USE_ET_PROXY: bool = False
    HTTP_PROXY: str = ""
    HTTPS_PROXY: str = ""
    NO_PROXY: str = ""

    # Neo4j settings
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_password"

    # PostgreSQL settings
    PG_MAJOR: int = 16
    POSTGRE_PORT: int = 5432
    POSTGRE_USER: str = "postgre_user"
    POSTGRE_PASSWORD: str = "postgre_password"
    POSTGRE_DB: str = "postgre_db"
    POSTGRE_HOST: str = "localhost"
    DJANGO_DB: str = "default"

    # Derived PostgreSQL settings
    DATABASE_URL: str = "postgresql://postgre_user:postgre_password@localhost:5432/postgre_db"

    # MLflow settings
    MLFLOW_USER: str = "mlflow_user"
    MLFLOW_PASSWORD: str = "mlflow_password"
    MLFLOW_DB: str = "mlflow_db"
    MLFLOW_PORT: int = 5002
    MLFLOW_TRACKING_URI: str = "http://mlflow:5002"
    MLFLOW_S3_ENDPOINT_URL: str = "http://minio:9000"
    MLFLOW_S3_IGNORE_TLS: bool = True

    # Derived MLflow settings
    MLFLOW_BACKEND_STORE_URI: str = "postgresql+psycopg2://postgre_user:postgre_password@localhost:5432/mlflow_db"
    MLFLOW_ARTIFACT_ROOT: str = "s3://minio:minio123@http://minio:9000/mlflow"

    # MinIO settings
    MINIO_PORT: int = 9000
    MINIO_CONSOLE_PORT: int = 9001
    MINIO_CLIENT_PORT: int = 9002
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio123"
    MINIO_ROOT_USER: str = "minio"
    MINIO_ROOT_PASSWORD: str = "minio123"
    MINIO_API_URL: str = "http://minio:9000"
    MINIO_URL: str = "http://minio:9000"

    # AWS settings for MinIO compatibility
    AWS_ACCESS_KEY_ID: str = "minio"
    AWS_SECRET_ACCESS_KEY: str = "minio123"
    AWS_DEFAULT_REGION: str = "eu-west-1"

    # Label Studio settings
    LABEL_STUDIO_USER: str = "labelstudio_user"
    LABEL_STUDIO_PASSWORD: str = "labelstudio_password"
    LABEL_STUDIO_DB: str = "labelstudio_db"
    LABEL_STUDIO_HOST: str = "label-studio"
    LABEL_STUDIO_PORT: int = 8081
    LABEL_STUDIO_USERNAME: str = "admin_user"
    LABEL_STUDIO_EMAIL: str = "admin@example.com"
    LABEL_STUDIO_API_KEY: str = "secure_api_key_123"
    LABEL_STUDIO_BUCKET_NAME: str = "mlflow-source"
    LABEL_STUDIO_BUCKET_PREFIX: str = "source_data/"
    LABEL_STUDIO_BUCKET_ENDPOINT_URL: str = "http://minio:9000"
    LABEL_STUDIO_BUCKET_ACCESS_KEY: str = "minio"
    LABEL_STUDIO_BUCKET_SECRET_KEY: str = "minio123"
    LABEL_STUDIO_TARGET_BUCKET: str = "mlflow-annotations"
    LABEL_STUDIO_TARGET_PREFIX: str = "annotations/"
    LABEL_STUDIO_TARGET_ACCESS_KEY: str = "minio"
    LABEL_STUDIO_TARGET_SECRET_KEY: str = "minio123"
    LABEL_STUDIO_TARGET_ENDPOINT_URL: str = "http://minio:9000"
    LABEL_STUDIO_PROJECT_NAME: str = "proj-1"
    LABEL_STUDIO_PROJECT_TITLE: str = "Machine Learning Annotations Project"
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED: bool = True
    LS_DATABASE_URL: str = "postgresql://labelstudio_user:labelstudio_password@localhost:5432/labelstudio_db"

    # Prometheus settings
    PROMETHEUS_PORT: int = 9090

    # GLiNER settings
    GLINER_BASIC_AUTH_USER: str = "my_user"
    GLINER_BASIC_AUTH_PASS: str = "my_password"
    GLINER_MODEL_NAME: str = "knowledgator/gliner-multitask-large-v0.5"
    LABEL_STUDIO_ML_BACKENDS: str = '[{"url": "http://gliner:9097", "name": "GLiNER"}]'

    GLIREL_MODEL_NAME: str = "jackboyla/glirel-large-v0"
    # Secret Key
    SECRET_KEY: str = "super_secret_key_123"

    # General settings
    WORKERS: int = 4
    THREADS: int = 4
    TEST_ENV: str = "my_test_env"
    LOCIP: str = "192.168.1.106"

    # ML Backend
    MLBACKEND_PORT: int = 9097

    # Default Models
    DEFAULT_MODELS: str = "urchade/gliner_smallv2.1"

    # Training Configuration
    TRAIN_CONFIG: dict = {
        "num_steps": 10_000,
        "train_batch_size": 2,
        "eval_every": 1_000,
        "save_directory": "checkpoints",
        "warmup_ratio": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr_encoder": 1e-5,
        "lr_others": 5e-5,
        "freeze_token_rep": False,
        "max_types": 25,
        "shuffle_types": True,
        "random_drop": True,
        "max_neg_type_ratio": 1,
        "max_len": 384,
    }

    # Text splitting settings
    TEXT_CHUNK_SIZE: int = 1000
    TEXT_CHUNK_OVERLAP: int = 200
    CONF_FILE: str = "../conf/gli_config.yml"
    

    # Ollama
    OLLAMA_MODEL: str ="nomic-embed-text"
    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Instanciation de la configuration
settings = Settings()

# Constants for Models and Device
MODELS = {
    "GLiNER-S": "urchade/gliner_smallv2.1",
    "GLiNER-M": "urchade/gliner_mediumv2.1",
    "GLiNER-L": "urchade/gliner_largev2.1",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----

# __init__.py
#-----

#-----

# concsrc3.py
#-----

#-----

# main.py
#-----
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

#-----

# dependencies.py
#-----
# dependencies.py

import yaml
from fastapi import Depends
from sqlalchemy.orm import Session
from typing import Generator

from config import settings
from utils.database import SessionLocal
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.document_processor import DocumentProcessor
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from services.rag_service import RAGChainService
from services.embedding_service import EmbeddingService
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_postgres import PGVector
from langchain_ollama.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase

# Dependency to get the SQLAlchemy session
def get_db() -> Generator[Session, None, None]:
    """Yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get the S3 service
def get_s3_service() -> S3Service:
    """Returns an instance of the S3 service."""
    return S3Service(
        s3_client=None,
        endpoint_url=settings.MINIO_URL,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        input_bucket="docs-input",
        output_bucket="docs-output",
        layouts_bucket="layouts"
    )

# Dependency to get the MLflow service
def get_mlflow_service() -> MLFlowService:
    """Returns an instance of the MLflow service."""
    return MLFlowService(tracking_uri=settings.MLFLOW_TRACKING_URI)

# Dependency for PGVector service
def get_pgvector_service() -> PGVectorService:
    return PGVectorService(
        db_url=settings.DATABASE_URL,
        table_name="document_embeddings"
    )


# Dependency for embedding service
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.OLLAMA_MODEL)


# Dependency for PGVector vector store
def get_pgvector_vector_store() -> PGVector:
    embedding_service = get_embedding_service()
    return PGVector(
        collection_name="document_embeddings",
        connection=settings.DATABASE_URL,
        embeddings=embedding_service.embedding_model.embed_documents
    )

# Dependency to get the Neo4j driver
def get_neo4j_driver() -> GraphDatabase:
    """Returns a Neo4j driver instance."""
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

# Dependency to get the Neo4j service
def get_neo4j_service() -> Neo4jService:
    """Returns an instance of the Neo4j service."""
    return Neo4jService(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )

# Initialize reusable text splitter
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns an instance of the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.TEXT_CHUNK_SIZE,
        chunk_overlap=settings.TEXT_CHUNK_OVERLAP
    )

# Dependency to get the GLiNER extractor
def get_gliner_extractor() -> GLiNERLinkExtractor:
    """Returns an instance of the GLiNER extractor."""
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GLiNERLinkExtractor(
        labels=config["labels"],
        model=settings.GLINER_MODEL
    )

# Dependency to get the graph transformer
def get_graph_transformer() -> GlinerGraphTransformer:
    """Returns an instance of the GlinerGraphTransformer."""
    with open(settings.CONF_FILE, 'r') as file:
        config = yaml.safe_load(file)
    return GlinerGraphTransformer(
        allowed_nodes=config["allowed_nodes"],
        allowed_relationships=config["allowed_relationships"],
        gliner_model=settings.GLINER_MODEL_NAME,
        glirel_model=settings.GLIREL_MODEL_NAME,
        entity_confidence_threshold=0.1,
        relationship_confidence_threshold=0.1,
    )

# Dependency to get the document processor
def get_document_processor(db: Session = Depends(get_db)) -> DocumentProcessor:
    """Returns an instance of the Document Processor."""
    s3_service = get_s3_service()
    mlflow_service = get_mlflow_service()
    pgvector_service = get_pgvector_service()
    text_splitter = get_text_splitter()
    graph_transformer = get_graph_transformer()
    neo4j_service = get_neo4j_service()

    return DocumentProcessor(
        s3_service=s3_service,
        mlflow_service=mlflow_service,
        pgvector_service=pgvector_service,
        neo4j_service=neo4j_service,
        session=db,
        text_splitter=text_splitter,
        graph_transformer=graph_transformer
    )

# Dependency to get the RAG service
def get_rag_service() -> RAGChainService:
    """Returns an instance of the RAG Chain Service."""
    vector_store = get_pgvector_vector_store()
    return RAGChainService(retriever=vector_store.as_retriever())

#-----

# utils/helpers.py
#-----
import os
import random
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Load a YAML configuration file
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Save data to a YAML file
def save_to_yaml(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a YAML file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

# Load a JSON configuration file
def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed content of the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to a JSON file
def save_to_json(data: Dict[str, Any], file_path: str):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Create a directory if it doesn't exist
def ensure_directory(directory_path: str):
    """
    Create a directory if it does not already exist.

    Args:
        directory_path (str): Path of the directory to create.
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

# Delete a directory
def delete_directory(directory_path: str):
    """
    Delete a directory and all its contents.

    Args:
        directory_path (str): Path of the directory to delete.
    """
    shutil.rmtree(directory_path, ignore_errors=True)

# Generate a random string
def generate_random_string(length: int = 8) -> str:
    """
    Generate a random alphanumeric string of specified length.

    Args:
        length (int): Length of the generated string (default is 8).

    Returns:
        str: Random alphanumeric string.
    """
    characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(characters, k=length))

# Get an environment variable with a default value
def get_env_variable(key: str, default: Optional[str] = None) -> str:
    """
    Get the value of an environment variable or return a default value if not set.

    Args:
        key (str): The environment variable key.
        default (str, optional): The default value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default value.
    """
    return os.getenv(key, default)

#-----

# utils/metrics.py
#-----
import psutil, GPUtil
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
#-----

# utils/logging_utils.py
#-----

# utils/logging.py
import os
from loguru import logger
from huggingface_hub import HfApi
import mlflow
from mlflow.tracking import MlflowClient
from codecarbon import EmissionsTracker
from typing import Optional
import time

class ModelLoggerService:
    def __init__(self):
        self.hf_api = HfApi()  # Initialize the Hugging Face API client
        self.huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
        self.client = MlflowClient()
        self.emissions_tracker = None

        # Initialize the MLflow tracking URI
        db_url = os.getenv("DATABASE_URL", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(db_url)

        # Initialize static models and CodeCarbon tracker
        self.static_models = {
            "Ollama Embedding Model": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(self.huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2")),
            "GLiNER Extractor Model": ("E3-JSI/gliner-multi-pii-domains-v1", os.path.join(self.huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1")),
            "Gliner Transformer Model": ("knowledgator/gliner-multitask-large-v0.5", os.path.join(self.huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5")),
            "Tokenizer Model": ("microsoft/deberta-v3-large", os.path.join(self.huggingface_cache, "models--microsoft--deberta-v3-large"))
        }
        self.initialize_emissions_tracker()

    def initialize_emissions_tracker(self):
        """
        Initialize CodeCarbon tracker with lock file cleanup.
        """
        lock_file = "/tmp/.codecarbon.lock"
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info("CodeCarbon lock file removed.")
            except Exception as e:
                logger.warning(f"Unable to remove CodeCarbon lock file: {e}")

        self.emissions_tracker = EmissionsTracker(project_name="model_logging", save_to_file=False, save_to_prometheus=True, prometheus_url="localhost:8002")
        logger.info("CodeCarbon tracker initialized.")

    def log_model_details(self):
        logger.info("Starting model logging process...")
        mlflow.end_run()  # Ensure no active runs are in progress

        try:
            with mlflow.start_run(run_name="Model Logging") as run:
                run_id = run.info.run_id

                for model_name, (model_id, model_file_path) in self.static_models.items():
                    logger.info(f"Processing model: {model_name}")
                    self._log_model_metadata(model_name, model_id, model_file_path, run_id)

            logger.info("Model logging process completed.")
            return {"message": "Model logging completed successfully"}
        except Exception as e:
            logger.error(f"Error in logging model details: {e}")
            return {"error": str(e)}

    def _log_model_metadata(self, model_name, model_id, model_file_path, run_id):
        try:
            # Check if the model is registered in MLflow
            registered_models = [rm.name for rm in self.client.search_registered_models()]
            if model_name not in registered_models:
                self.client.create_registered_model(model_name)

            # Fetch model metadata from Hugging Face
            model_info = self.hf_api.model_info(model_id)
            model_version = model_info.sha  # Unique identifier for version
            model_description = self._fetch_readme(model_id) or "No description available."  # Use README.md content as description
            model_tags = model_info.tags

            # Log metadata to MLflow
            mlflow.set_tag(f"{model_name}_description", model_description)
            for tag in model_tags:
                mlflow.set_tag(f"{model_name}_tag_{tag}", True)
            mlflow.log_param(f"{model_name}_version", model_version)

            # Log model file as artifact if it exists
            if os.path.exists(model_file_path):
                artifact_path = f"artifacts/{model_name}"
                mlflow.log_artifact(model_file_path, artifact_path=artifact_path)
                self.client.create_model_version(
                    name=model_name,
                    source=f"{mlflow.get_artifact_uri()}/{artifact_path}",
                    run_id=run_id
                )
            else:
                logger.warning(f"Model path not found: {model_file_path}")
        except Exception as e:
            logger.error(f"Error logging metadata for model {model_name}: {e}")

    def _fetch_readme(self, model_id: str) -> Optional[str]:
        """
        Fetch the README.md content of a Hugging Face model to use as a description.
        """
        try:
            readme_content = self.hf_api.model_info(model_id).cardData.get("model_card", "")
            return readme_content
        except Exception as e:
            logger.warning(f"Unable to fetch README.md for model {model_id}: {e}")
            return None

    def log_query(self, query: str):
        try:
            with mlflow.start_run(run_name="Query Logging") as run:
                mlflow.log_param("query", query)
                mlflow.log_param("timestamp", time.time())
                logger.info("Query logged successfully.")
                return {"message": "Query logged successfully"}
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            return {"error": str(e)}

#-----

# utils/database.py
#-----
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from config import settings
import logging

# Load database URL from settings
db_url = settings.DATABASE_URL

# Create the SQLAlchemy engine
engine = create_engine(db_url, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for the models
Base = declarative_base()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Context manager for database sessions
@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

# Create all tables
def init_db():
    from sqlalchemy import text  # To execute raw SQL if needed
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

#-----

# models/pydantic/entity.py
#-----
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EntityBase(BaseModel):
    name: str
    type: str
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }

class EntityCreate(EntityBase):
    """
    Model for creating a new entity.
    Inherits from EntityBase and can be extended for additional fields required at creation.
    """
    pass

class Entity(EntityBase):
    """
    Model representing an entity with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the entity")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "name": "Sample Entity",
                "type": "Organization",
                "properties": {
                    "location": "New York",
                    "employees": 100
                }
            }
        }

#-----

# models/pydantic/relationship.py
#-----
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class RelationshipBase(BaseModel):
    source_id: str = Field(..., description="The ID of the source entity")
    target_id: str = Field(..., description="The ID of the target entity")
    type: str = Field(..., description="The type of the relationship")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }

class RelationshipCreate(RelationshipBase):
    """
    Model for creating a new relationship.
    Inherits from RelationshipBase and can be extended for additional fields required at creation.
    """
    pass

class Relationship(RelationshipBase):
    """
    Model representing a relationship with an ID, as returned from the database or API.
    """
    id: str = Field(..., description="The unique identifier of the relationship")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "456e4567-e89b-12d3-a456-426614174003",
                "source_id": "123e4567-e89b-12d3-a456-426614174001",
                "target_id": "789e4567-e89b-12d3-a456-426614174002",
                "type": "Partnership",
                "properties": {
                    "since": "2021-01-01",
                    "status": "active"
                }
            }
        }

#-----

# models/pydantic/document.py
#-----
from pydantic import BaseModel
from typing import Dict, Any

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Sample Document",
                "content": "This is the content of the document.",
                "metadata": {
                    "author": "John Doe",
                    "tags": ["example", "sample"]
                },
                "created_at": "2023-11-12T10:00:00Z",
                "updated_at": "2023-11-12T12:00:00Z"
            }
        }

#-----

# models/sqlalchemy/document_log.py
#-----
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Callable
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)

    def __repr__(self):
        return f"<DocumentLog(id={self.id}, file_name='{self.file_name}', s3_url='{self.s3_url}')>"


class DocumentLogService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def log_document(self, file_name: str, s3_url: str) -> None:
        """Logs a document entry in the database."""
        try:
            with self.session_factory() as session:
                log_entry = DocumentLog(file_name=file_name, s3_url=s3_url)
                session.add(log_entry)
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"Failed to log document: {e}")

#-----

# middleware/custom_metrics.py
#-----
import time
import psutil
import GPUtil
from prometheus_client import Histogram, Counter, Gauge
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests received")
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds")
CPU_USAGE = Gauge("app_cpu_usage_percent", "CPU usage in percent")
MEMORY_USAGE = Gauge("app_memory_usage_bytes", "Memory usage in bytes")
GPU_MEMORY_USAGE = Gauge("app_gpu_memory_usage_bytes", "GPU memory usage in bytes")

class CustomMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        REQUEST_COUNT.inc()  # Increment the request count

        # Call the next middleware or actual request handler
        response = await call_next(request)

        # Measure latency
        process_time = time.time() - start_time
        REQUEST_LATENCY.observe(process_time)

        # Log system metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)

        # Log GPU metrics if available
        gpus = GPUtil.getGPUs()
        if gpus:
            GPU_MEMORY_USAGE.set(gpus[0].memoryUsed)  # Only log the first GPU

        return response
#-----

# routers/entities.py
#-----
# routers/entities.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.pydantic.entity import EntityCreate, Entity
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/entities/", response_model=Entity)
async def create_entity(entity: EntityCreate):
    logger.info(f"Creating entity: {entity.name}")
    try:
        created_entity = neo4j_service.create_entity(entity)
        logger.info(f"Successfully created entity: {created_entity.name}")
        return created_entity
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/", response_model=List[Entity])
async def get_entities():
    logger.info("Retrieving all entities")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(entity_id: str):
    logger.info(f"Retrieving entity with ID: {entity_id}")
    try:
        entity = neo4j_service.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully retrieved entity: {entity.name}")
        return entity
    except Exception as e:
        logger.error(f"Error retrieving entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity_id}", response_model=dict)
async def delete_entity(entity_id: str):
    logger.info(f"Deleting entity with ID: {entity_id}")
    try:
        success = neo4j_service.delete_entity(entity_id)
        if not success:
            raise HTTPException(status_code=404, detail="Entity not found")
        logger.info(f"Successfully deleted entity with ID: {entity_id}")
        return {"message": f"Entity {entity_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/graph.py
#-----
# routers/graph.py
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List, Dict, Any

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.pydantic.entity import Entity
from models.pydantic.relationship import Relationship

router = APIRouter()

neo4j_service: Neo4jService = get_neo4j_service()


@router.get("/entities/", response_model=List[Entity])
async def get_all_entities():
    logger.info("Retrieving all entities from the graph")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities from the graph")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/", response_model=List[Relationship])
async def get_all_relationships():
    logger.info("Retrieving all relationships from the graph")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships from the graph")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/", response_model=Dict[str, List[Dict[str, Any]]])
async def visualize_graph():
    logger.info("Generating graph visualization data")
    try:
        graph_data = neo4j_service.generate_graph_visualization()
        logger.info("Graph visualization data generated successfully")
        return graph_data
    except Exception as e:
        logger.error(f"Error generating graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/relationships.py
#-----
# # routers/relationships.py
from fastapi import APIRouter, HTTPException
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from models.pydantic.relationship import RelationshipCreate, Relationship
from dependencies import get_neo4j_service

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.post("/relationships/", response_model=Relationship)
async def create_relationship(relationship: RelationshipCreate):
    logger.info(f"Creating relationship: {relationship.type} between {relationship.source_id} and {relationship.target_id}")
    try:
        created_relationship = neo4j_service.create_relationship(relationship)
        logger.info(f"Successfully created relationship: {created_relationship.type}")
        return created_relationship
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/", response_model=List[Relationship])
async def get_relationships():
    logger.info("Retrieving all relationships")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/{relationship_id}", response_model=Relationship)
async def get_relationship(relationship_id: str):
    logger.info(f"Retrieving relationship with ID: {relationship_id}")
    try:
        relationship = neo4j_service.get_relationship(relationship_id)
        if not relationship:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully retrieved relationship: {relationship.type}")
        return relationship
    except Exception as e:
        logger.error(f"Error retrieving relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/relationships/{relationship_id}", response_model=dict)
async def delete_relationship(relationship_id: str):
    logger.info(f"Deleting relationship with ID: {relationship_id}")
    try:
        success = neo4j_service.delete_relationship(relationship_id)
        if not success:
            raise HTTPException(status_code=404, detail="Relationship not found")
        logger.info(f"Successfully deleted relationship with ID: {relationship_id}")
        return {"message": f"Relationship {relationship_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/logging_router.py
#-----
# routers/logging.py
from fastapi import APIRouter
from utils.logging_utils import ModelLoggerService

router = APIRouter()

# Initialize the ModelLoggerService
model_logger_service = ModelLoggerService()

@router.post("/log_models/")
def log_models():
    """API endpoint to trigger logging of model details."""
    return model_logger_service.log_model_details()

@router.post("/log_queries/")
def log_queries(query: str):
    """API endpoint to trigger logging of queries."""
    return model_logger_service.log_query(query)
#-----

# routers/documents.py
#-----
# routers/documents.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
from typing import List
from loguru import logger

from services.document_processor import DocumentProcessor
from services.rag_service import RAGChainService
from dependencies import get_document_processor, get_rag_service

router = APIRouter()

# Dependency injection
document_processor: DocumentProcessor = get_document_processor()
rag_service: RAGChainService = get_rag_service()


@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False),
):
    """
    Upload and process documents for storage and feature extraction.
    """
    logger.info(f"Received {len(files)} files for upload")
    success_count, partial_success_count, failure_count = 0, 0, 0

    for file in files:
        temp_file = Path(f"/tmp/{file.filename}")
        async with temp_file.open("wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Process and log the document
        try:
            result = document_processor.process_file(
                temp_file, use_ocr, export_figures, export_tables, enrich_figures
            )
            counts = document_processor.export_document(
                result, temp_file.parent, export_formats, export_figures, export_tables
            )
            success_count += counts[0]
            partial_success_count += counts[1]
            failure_count += counts[2]
        except Exception as e:
            logger.error(f"Error processing document {file.filename}: {e}")
            failure_count += 1
        finally:
            temp_file.unlink()

    return {
        "message": "Documents processed and stored successfully",
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count,
    }


@router.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    """
    Index a single document by extracting entities and relationships.
    """
    logger.info(f"Indexing document: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    # Use aiofiles for asynchronous file writing
    import aiofiles
    async with aiofiles.open(temp_file, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        document_processor(temp_file)
        logger.info(f"Successfully indexed document: {file.filename}")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        temp_file.unlink()



@router.post("/rag_process/")
async def process_rag_document(file: UploadFile = File(...)):
    """
    Process a document for RAG, splitting it, embedding it, and storing it in the vector store.
    """
    logger.info(f"Processing document for RAG: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")

    async with temp_file.open("wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        result = rag_service.process_document_for_rag(temp_file)
        logger.info(f"Document successfully processed for RAG: {file.filename}")
        return {"message": "Document successfully processed for RAG.", "details": result}
    except Exception as e:
        logger.error(f"Error processing document for RAG: {file.filename}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document for RAG: {e}")
    finally:
        temp_file.unlink()

#-----

# routers/search.py
#-----
from fastapi import APIRouter, HTTPException, Query
from typing import List
from loguru import logger

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.pydantic.entity import Entity
from models.pydantic.relationship import Relationship

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.get("/search/entities/", response_model=List[Entity])
async def search_entities(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching entities with keyword: {keyword}")
    try:
        entities = neo4j_service.search_entities(keyword)
        logger.info(f"Found {len(entities)} entities matching keyword: {keyword}")
        return entities
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/relationships/", response_model=List[Relationship])
async def search_relationships(keyword: str = Query(..., description="Keyword to search for")):
    logger.info(f"Searching relationships with keyword: {keyword}")
    try:
        relationships = neo4j_service.search_relationships(keyword)
        logger.info(f"Found {len(relationships)} relationships matching keyword: {keyword}")
        return relationships
    except Exception as e:
        logger.error(f"Error searching relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# services/neo4j_service.py
#-----
from neo4j import GraphDatabase, Transaction
from loguru import logger
from typing import Dict, Any, Optional, List


class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        with self.driver.session() as session:
            return session.write_transaction(self._create_node_transaction, label, properties)

    @staticmethod
    def _create_node_transaction(tx: Transaction, label: str, properties: Dict[str, Any]) -> Optional[int]:
        query = f"""
        CREATE (n:{label} $properties)
        RETURN id(n) AS node_id
        """
        try:
            result = tx.run(query, properties=properties)
            node_id = result.single()["node_id"]
            logger.info(f"Node created with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return None

    def create_relationship(self, source_id: int, target_id: int, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        with self.driver.session() as session:
            return session.write_transaction(
                self._create_relationship_transaction, source_id, target_id, relationship_type, properties
            )

    @staticmethod
    def _create_relationship_transaction(tx: Transaction, source_id: int, target_id: int, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $source_id AND id(b) = $target_id
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r
        """
        try:
            result = tx.run(query, source_id=source_id, target_id=target_id, properties=properties or {})
            if result.single():
                logger.info(f"Relationship {relationship_type} created between nodes {source_id} and {target_id}")
                return True
            else:
                logger.error("Failed to create relationship: No result returned")
                return False
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False

    def get_all_entities(self) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_entities_transaction)

    @staticmethod
    def _get_all_entities_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        query = """
        MATCH (e)
        RETURN id(e) AS id, labels(e) AS labels, properties(e) AS properties
        """
        try:
            result = tx.run(query)
            entities = [{"id": record["id"], "labels": record["labels"], "properties": record["properties"]} for record in result]
            return entities
        except Exception as e:
            logger.error(f"Failed to retrieve entities: {e}")
            return []

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_relationships_transaction)

    @staticmethod
    def _get_all_relationships_transaction(tx: Transaction) -> List[Dict[str, Any]]:
        query = """
        MATCH ()-[r]->()
        RETURN id(r) AS id, type(r) AS type, startNode(r) AS source, endNode(r) AS target, properties(r) AS properties
        """
        try:
            result = tx.run(query)
            relationships = [
                {
                    "id": record["id"],
                    "type": record["type"],
                    "source": record["source"],
                    "target": record["target"],
                    "properties": record["properties"],
                }
                for record in result
            ]
            return relationships
        except Exception as e:
            logger.error(f"Failed to retrieve relationships: {e}")
            return []

    def generate_graph_visualization(self) -> dict:
        with self.driver.session() as session:
            nodes = session.read_transaction(self._get_all_entities_transaction)
            relationships = session.read_transaction(self._get_all_relationships_transaction)
            return {"nodes": nodes, "relationships": relationships}

#-----

# services/s3_service.py
#-----
# services/s3_service.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from loguru import logger
from typing import Optional
from pathlib import Path

class S3Service:
    def __init__(self, s3_client, endpoint_url: str, access_key: str, secret_key: str, region_name: Optional[str] = None, input_bucket: str = "input", output_bucket: str = "output", layouts_bucket: str = "layouts"):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.layouts_bucket = layouts_bucket

        logger.info(f"Connected to S3 at {endpoint_url}")

    def upload_file(self, file_path: Path, bucket_name: str, object_name: Optional[str] = None) -> Optional[str]:
        if object_name is None:
            object_name = file_path.name
        try:
            self.s3_client.upload_file(str(file_path), bucket_name, object_name)
            logger.info(f"File {file_path} uploaded to bucket {bucket_name} as {object_name}")
            return f"s3://{bucket_name}/{object_name}"
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to S3: {e}")
        return None

    def download_file(self, bucket_name: str, object_name: str, download_path: Path) -> bool:
        try:
            self.s3_client.download_file(bucket_name, object_name, str(download_path))
            logger.info(f"File {object_name} downloaded from bucket {bucket_name} to {download_path}")
            return True
        except NoCredentialsError:
            logger.error("Credentials not available for S3.")
        except ClientError as e:
            logger.error(f"Failed to download file {object_name} from S3: {e}")
        return False

    def create_bucket(self, bucket_name: str) -> bool:
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} created successfully.")
            return True
        except ClientError as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
        return False

    def list_buckets(self) -> Optional[list]:
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            logger.info(f"Buckets retrieved: {buckets}")
            return buckets
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
        return None

    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} deleted from bucket {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file {object_name} from bucket {bucket_name}: {e}")
        return False

    def file_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"File {object_name} exists in bucket {bucket_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"File {object_name} does not exist in bucket {bucket_name}")
            else:
                logger.error(f"Error checking existence of file {object_name} in bucket {bucket_name}: {e}")
        return False

#-----

# services/rag_service.py
#-----
# service/rag_service.py
import os
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

class RAGChainService:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._initialize_llm()

        # Define the prompt
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
        )

    def _initialize_llm(self):
        HF_API_KEY = os.environ.get("HF_API_KEY")
        HF_LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        return HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL_ID,
            huggingfacehub_api_token=HF_API_KEY,
        )

    def format_docs(self, docs: Iterable[LCDocument]):
        """
        Format the documents for RAG.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        """
        Build the RAG chain.
        """
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run_query(self, query: str):
        """
        Run the query through the RAG chain.
        """
        rag_chain = self.build_chain()
        return rag_chain.invoke(query)

#-----

# services/mlflow_service.py
#-----
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path
import os
import json
from typing import Dict, Any

class MLFlowService:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    def start_run(self, run_name: str):
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run_name}")

    def log_params(self, params: Dict[str, Any]):
        try:
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, Any]):
        try:
            mlflow.log_metrics(metrics)
            logger.info(f"Logged metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, file_path: str, artifact_path: str = None):
        try:
            mlflow.log_artifact(file_path, artifact_path)
            logger.info(f"Logged artifact: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact to MLflow: {e}")

    def register_model(self, model_name: str, model_dir: Path):
        try:
            model_uri = f"{self.tracking_uri}/{model_dir}"
            self.client.create_registered_model(model_name)
            self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id
            )
            logger.info(f"Model {model_name} registered successfully.")
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")

    def get_model_version(self, model_name: str):
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            logger.info(f"Retrieved versions for model {model_name}: {versions}")
            return versions
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return None

    def download_model(self, model_name: str, version: str, download_dir: str):
        try:
            model_uri = f"models:/{model_name}/{version}"
            local_path = mlflow.pyfunc.load_model(model_uri).save(download_dir)
            logger.info(f"Model {model_name} version {version} downloaded successfully to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download model {model_name} version {version}: {e}")
            return None

    def list_registered_models(self):
        try:
            models = self.client.list_registered_models()
            logger.info(f"Retrieved registered models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

    def set_tracking_uri(self, uri: str):
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI updated to: {uri}")

#-----

# services/pgvector_service.py
#-----
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, List, Optional
from loguru import logger


class PGVectorService:
    def __init__(self, db_url: str, table_name: str = "document_vectors"):
        """
        Initialize PGVectorService with a PostgreSQL connection and table name.

        Args:
            db_url (str): Database connection string.
            table_name (str): Name of the table to store and query vectors.
        """
        self.db_url = db_url
        self.table_name = table_name
        self.connection = self._connect_to_db()
        self.cursor = self.connection.cursor()
        self._ensure_table_exists()

    def _connect_to_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            connection = psycopg2.connect(self.db_url)
            logger.info("Successfully connected to PostgreSQL database.")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _ensure_table_exists(self):
        """Ensure the required table exists in the database."""
        try:
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding VECTOR,
                    metadata JSONB,
                    content TEXT
                );
            """)
            self.connection.commit()
            logger.info(f"Table '{self.table_name}' ensured in database.")
        except Exception as e:
            logger.error(f"Failed to ensure table exists: {e}")
            raise

    def store_vector(self, embedding: List[float], metadata: Dict[str, Any], content: str) -> Optional[int]:
        """
        Store a vector in the database.

        Args:
            embedding (List[float]): Vector embedding.
            metadata (Dict[str, Any]): Metadata for the document.
            content (str): Document content.

        Returns:
            Optional[int]: Row ID of the stored vector.
        """
        try:
            self.cursor.execute(
                f"""
                INSERT INTO {self.table_name} (embedding, metadata, content)
                VALUES (%s, %s, %s) RETURNING id;
                """,
                (embedding, Json(metadata), content)
            )
            row_id = self.cursor.fetchone()[0]
            self.connection.commit()
            logger.info(f"Vector stored with ID {row_id}.")
            return row_id
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            self.connection.rollback()
            return None

    def search_vector(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the nearest vectors.

        Args:
            query_vector (List[float]): Query vector.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict[str, Any]]: List of results with metadata and distances.
        """
        try:
            self.cursor.execute(
                f"""
                SELECT id, content, metadata, embedding <=> %s AS distance
                FROM {self.table_name}
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (query_vector, k)
            )
            results = self.cursor.fetchall()
            logger.info(f"Found {len(results)} nearest vectors.")
            return [
                {"id": row[0], "content": row[1], "metadata": row[2], "distance": row[3]}
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error searching vector: {e}")
            return []

    def close(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

#-----

# services/model_manager.py
#-----
import torch
from typing import Dict, Any
from gliner import GLiNER
from transformers import AutoTokenizer
from services.s3_service import S3Service
from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

class ModelManager:
    def __init__(self, s3_service: S3Service):
        self.s3_service = s3_service
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracker_active = False
        self.emissions_tracker = None
        self.mlflow_client = MlflowClient()
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_name: str):
        model_path = Path("models") / model_name
        if model_path.exists():
            model = GLiNER.from_pretrained(str(model_path)).to(self.device)
            return model
        elif model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(model_name).to(self.device)
            model.save_pretrained(model_path)
            return model
        else:
            raise ValueError(f"Model {model_name} not found.")

    def log_model_metrics(self, metrics: Dict[str, Any]):
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")

    def process_model(self, model_name: str, inputs: Dict[str, Any]):
        # Ensure any previous MLflow run is ended before starting a new one
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"Processing {model_name}"):
            # Initialize CodeCarbon tracker if none is active
            if not self.tracker_active:
                try:
                    self.emissions_tracker = EmissionsTracker(project_name="model_processing")
                    self.emissions_tracker.start()
                    self.tracker_active = True
                except Exception as e:
                    logger.warning(f"Unable to start CodeCarbon: {e}")
                    self.emissions_tracker = None

            # Load the model
            model = self.load_model(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs_tokenized = tokenizer(inputs["text"], return_tensors="pt").to(self.device)

            # Run inference
            output = model(**inputs_tokenized)

            # Capture emissions if CodeCarbon tracker is active
            emissions = None
            if self.emissions_tracker and self.tracker_active:
                try:
                    emissions = self.emissions_tracker.stop()
                except Exception as e:
                    logger.warning(f"Error stopping CodeCarbon tracker: {e}")
                finally:
                    self.tracker_active = False  # Reset for next use

            # Log hardware resource usage
            metrics = {
                "gpu_memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cpu_usage": torch.get_num_threads(),
            }
            if emissions is not None:
                metrics["carbon_emissions"] = emissions

            self.log_model_metrics(metrics)

            return output

    def zip_and_upload_model(self, model_name: str):
        model_path = Path("models") / model_name
        zip_path = model_path.with_suffix(".zip")

        if not model_path.exists():
            raise ValueError(f"Model directory {model_name} does not exist.")

        # Create zip file of the model directory
        try:
            import shutil
            shutil.make_archive(str(model_path), 'zip', str(model_path))
        except Exception as e:
            logger.error(f"Failed to create zip archive for {model_name}: {e}")
            return None

        # Upload to S3 bucket
        s3_url = self.s3_service.upload_file(zip_path, bucket_name=self.s3_service.output_bucket)
        if s3_url:
            logger.info(f"Model {model_name} uploaded successfully to {s3_url}")
            return s3_url
        else:
            logger.error(f"Failed to upload model {model_name} to S3")
            return None

#-----

# services/document_processor.py
#-----
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.pgvector_service import PGVectorService
from services.neo4j_service import Neo4jService
from sqlalchemy.orm import Session
from models.sqlalchemy.document_log import DocumentLog
from loguru import logger
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer


class CustomPdfPipelineOptions(PdfPipelineOptions):
    """Custom pipeline options for PDF processing."""
    do_picture_classifier: bool = False


class DocumentProcessor:
    """Orchestrates the entire document processing pipeline: splitting, exporting, and indexing."""

    def __init__(
        self,
        s3_service: S3Service,
        mlflow_service: MLFlowService,
        pgvector_service: PGVectorService,
        neo4j_service: Neo4jService,
        session: Session,
        text_splitter: CharacterTextSplitter,
        graph_transformer: GlinerGraphTransformer,
    ):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.pgvector_service = pgvector_service
        self.neo4j_service = neo4j_service
        self.session = session
        self.text_splitter = text_splitter
        self.graph_transformer = graph_transformer

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool) -> DocumentConverter:
        """Create and configure a document converter."""
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)}
        )

    def log_document(self, file_name: str, s3_url: str):
        """Log document metadata into the database."""
        try:
            log = DocumentLog(file_name=file_name, s3_url=s3_url)
            self.session.add(log)
            self.session.commit()
            logger.info(f"Document logged: {file_name}")
        except Exception as e:
            logger.error(f"Failed to log document {file_name}: {e}")
            self.session.rollback()
            raise

    def export_document(
        self,
        result: ConversionResult,
        output_dir: Path,
        export_formats: List[str],
        export_figures: bool,
        export_tables: bool
    ):
        """Export document into the specified formats and upload to S3."""
        try:
            doc_filename = result.input.file.stem
            if result.status == ConversionStatus.SUCCESS:
                self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename)
                logger.info(f"Document exported successfully: {doc_filename}")
            else:
                logger.warning(f"Document export failed for {doc_filename}: {result.status}")
        except Exception as e:
            logger.error(f"Error exporting document: {e}")
            raise

    def _export_file(self, result, output_dir, export_formats, export_figures, export_tables, doc_filename):
        """Save and upload the exported document files."""
        for ext in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, ext, export_format=ext)

        if export_figures:
            self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket)
        if export_tables:
            self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket)

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json"):
        """Save a specific document format locally and upload it to S3."""
        file_path = output_dir / f"{doc_filename}.{ext}"
        with file_path.open("w", encoding="utf-8") as file:
            if export_format == "json":
                json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
            elif export_format == "yaml":
                yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
            elif export_format == "md":
                file.write(result.document.export_to_markdown())
        self.s3_service.upload_file(file_path, self.s3_service.output_bucket)

    def _export_images(self, result, figures_dir, doc_filename, bucket):
        """Export and upload document images."""
        figures_dir.mkdir(exist_ok=True)
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                self.s3_service.upload_file(image_path, bucket)

    def _export_tables(self, result, tables_dir, doc_filename, bucket):
        """Export and upload document tables."""
        tables_dir.mkdir(exist_ok=True)
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            self.s3_service.upload_file(csv_path, bucket)

    def process_and_index_document(self, document: Document):
        """Process a document: split, index into PGVector, and index into Neo4j."""
        try:
            logger.info(f"Processing document: {document.metadata.get('name', 'Unknown')}")

            # Step 1: Split the document into chunks
            split_docs = self.text_splitter.split_documents([document])
            logger.info(f"Document split into {len(split_docs)} chunks.")

            # Step 2: Index chunks into PGVector
            self.pgvector_service.index_documents(split_docs)

            # Step 3: Transform and index graph data into Neo4j
            graph_docs = self.graph_transformer.convert_to_graph_documents(split_docs)
            for graph_doc in graph_docs:
                self.neo4j_service.index_graph(graph_doc.nodes, graph_doc.edges)

            logger.info(f"Document indexed successfully: {document.metadata.get('name', 'Unknown')}")

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

#-----

# services/embedding_service.py
#-----
# services/embedding_service.py
from langchain_ollama.embeddings import OllamaEmbeddings
from loguru import logger
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str):
        self.embedding_model = OllamaEmbeddings(model=model_name)
        logger.info(f"Embedding model '{model_name}' initialized.")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.embed_query(text)
            logger.info(f"Generated embedding for the given text.")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_query(text)
#-----
