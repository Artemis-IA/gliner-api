
# config.py
#-----
from pydantic import BaseSettings

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_password"
    MINIO_URL: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "minio123"
    POSTGRES_USER: str = "postgres_user"
    POSTGRES_PASSWORD: str = "postgres_password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_DB: str = "postgres_db"
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    MLFLOW_TRACKING_URI: str = DATABASE_URL
    MLFLOW_ARTIFACT_URI: str = f"s3://{MINIO_ACCESS_KEY}:{MINIO_SECRET_KEY}@{MINIO_URL}/mlflow"
    ALLOWED_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()
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
from fastapi import Depends
from sqlalchemy.orm import Session
from utils.database import SessionLocal

# Dependency to get the SQLAlchemy session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Example of other shared dependencies
async def common_parameters(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

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

# utils/logging.py
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
from utils.config import DATABASE_URL



class ModelLoggerService:
    def __init__(self, db_url: str):
        mlflow.set_tracking_uri(db_url)
        self.client = MlflowClient()
        self.hf_api = HfApi()  # Initialize the Hugging Face API client
        self.huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
        self.static_models = {
            "Ollama Embedding Model": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(self.huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2")),
            "GLiNER Extractor Model": ("E3-JSI/gliner-multi-pii-domains-v1", os.path.join(self.huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1")),
            "Gliner Transformer Model": ("knowledgator/gliner-multitask-large-v0.5", os.path.join(self.huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5")),
            "Tokenizer Model": ("microsoft/deberta-v3-large", os.path.join(self.huggingface_cache, "models--microsoft--deberta-v3-large"))
        }

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

model_logger_service = ModelLoggerService(db_url=DATABASE_URL)

#-----

# utils/metrics.py
#-----
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics for Prometheus
REQUEST_COUNT = Counter("app_request_count", "Total number of requests received")
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds")
CPU_USAGE = Gauge("app_cpu_usage_percent", "CPU usage in percent")
MEMORY_USAGE = Gauge("app_memory_usage_bytes", "Memory usage in bytes")
GPU_MEMORY_USAGE = Gauge("app_gpu_memory_usage_bytes", "GPU memory usage in bytes")
CARBON_EMISSIONS = Gauge("app_carbon_emissions_grams", "Estimated carbon emissions in grams")

# Function to start the Prometheus metrics server
def start_metrics_server(port: int = 8002):
    """
    Start the Prometheus metrics server to expose application metrics.

    Args:
        port (int): The port to expose metrics on (default is 8002).
    """
    start_http_server(port)
    REQUEST_COUNT.inc()  # Increment the request count to indicate the server has started

#-----

# models/pydantic/entity.py
#-----
from pydantic import BaseModel
from typing import Dict, Any

class Entity(BaseModel):
    id: str
    name: str
    type: str
    properties: Dict[str, Any]

    class Config:
        schema_extra = {
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
from pydantic import BaseModel
from typing import Dict, Any

class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]

    class Config:
        schema_extra = {
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
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)

    def __repr__(self):
        return f"<DocumentLog(id={self.id}, file_name='{self.file_name}', s3_url='{self.s3_url}')>"

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
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List

from services.neo4j_service import Neo4jService
from dependencies import get_neo4j_service
from models.pydantic.entity import Entity
from models.pydantic.relationship import Relationship

router = APIRouter()

# Dependency injection
neo4j_service: Neo4jService = get_neo4j_service()

@router.get("/graph/entities/", response_model=List[Entity])
async def get_all_entities():
    logger.info("Retrieving all entities from the graph")
    try:
        entities = neo4j_service.get_all_entities()
        logger.info(f"Retrieved {len(entities)} entities from the graph")
        return entities
    except Exception as e:
        logger.error(f"Error retrieving entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/relationships/", response_model=List[Relationship])
async def get_all_relationships():
    logger.info("Retrieving all relationships from the graph")
    try:
        relationships = neo4j_service.get_all_relationships()
        logger.info(f"Retrieved {len(relationships)} relationships from the graph")
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/visualize/", response_model=dict)
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

# routers/logging.py
#-----
# routers/logging.py
from fastapi import APIRouter
from services.logging import ModelLoggerService

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
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from pathlib import Path
from loguru import logger

from services.document_processor import DocumentProcessor
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from dependencies import get_s3_service, get_document_processor, get_mlflow_service

router = APIRouter()

# Dependency injection
s3_service: S3Service = get_s3_service()
document_processor: DocumentProcessor = get_document_processor()
mlflow_service: MLFlowService = get_mlflow_service()

@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False)
):
    logger.info(f"Received {len(files)} files for upload")
    success_count, partial_success_count, failure_count = 0, 0, 0

    for file in files:
        temp_file = Path(f"/tmp/{file.filename}")
        with temp_file.open("wb") as out_file:
            content = await file.read()
            out_file.write(content)

        input_s3_url = s3_service.upload_file(temp_file, s3_service.input_bucket)
        document_processor.log_document(file.filename, input_s3_url)

        result = await document_processor.process_document(temp_file, use_ocr, export_figures, export_tables, enrich_figures)
        if result:
            counts = document_processor.export_document(result, export_formats, export_figures, export_tables)
            success_count += counts[0]
            partial_success_count += counts[1]
            failure_count += counts[2]

    return {
        "message": "Documents processed and stored successfully",
        "uploaded_to": s3_service.output_bucket,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count
    }

@router.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    logger.info(f"Indexing document: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")
    with temp_file.open("wb") as out_file:
        content = await file.read()
        out_file.write(content)

    try:
        document_processor.index_document(temp_file)
        logger.info(f"Successfully indexed document: {file.filename}")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
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
from typing import Dict, Any, Optional

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
            result = session.write_transaction(self._create_node_transaction, label, properties)
            return result

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
            success = session.write_transaction(self._create_relationship_transaction, source_id, target_id, relationship_type, properties)
            return success

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

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        with self.driver.session() as session:
            node = session.read_transaction(self._get_node_transaction, node_id)
            return node

    @staticmethod
    def _get_node_transaction(tx: Transaction, node_id: int) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        RETURN properties(n) AS properties
        """
        try:
            result = tx.run(query, node_id=node_id)
            record = result.single()
            if record:
                logger.info(f"Node retrieved with ID: {node_id}")
                return record["properties"]
            else:
                logger.error(f"Node with ID {node_id} not found")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve node: {e}")
            return None

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        with self.driver.session() as session:
            result = session.run(query, **(parameters or {}))
            return result.data()

#-----

# services/s3_service.py
#-----
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from loguru import logger
from typing import Optional
from pathlib import Path

class S3Service:
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, region_name: Optional[str] = None):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
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
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from sqlalchemy.orm import Session
from models.sqlalchemy.document_log import DocumentLog

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False

class DocumentProcessor:
    def __init__(self, s3_service: S3Service, mlflow_service: MLFlowService, session: Session):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.session = session

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool):
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
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        self.session.add(log)
        self.session.commit()

    def export_document(self, result: ConversionResult, output_dir: Path, export_formats: List[str], export_figures: bool, export_tables: bool):
        success_count, partial_success_count, failure_count = 0, 0, 0
        doc_filename = result.input.file.stem

        if result.status == ConversionStatus.SUCCESS:
            success_count += 1
            self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename)
        elif result.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
        else:
            failure_count += 1

        return success_count, partial_success_count, failure_count

    def _export_file(self, result, output_dir, export_formats: List[str], export_figures: bool, export_tables: bool, doc_filename: str):
        if "json" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "json", export_format="json")
        if "yaml" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "yaml", export_format="yaml")
        if "md" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "md", export_format="md")

        if export_figures:
            self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket)
        if export_tables:
            self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket)

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json"):
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
        figures_dir.mkdir(exist_ok=True)
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                self.s3_service.upload_file(image_path, bucket)

    def _export_tables(self, result, tables_dir, doc_filename, bucket):
        tables_dir.mkdir(exist_ok=True)
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            self.s3_service.upload_file(csv_path, bucket)

            html_path = tables_dir / f"{doc_filename}_table_{idx + 1}.html"
            with html_path.open("w", encoding="utf-8") as html_file:
                html_file.write(table.export_to_html())
            self.s3_service.upload_file(html_path, bucket)

#-----

# services/embedding_service.py
#-----
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

#-----
