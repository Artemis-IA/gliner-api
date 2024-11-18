import os
import re
import json
import random
import shutil
import zipfile
import time
import yaml
import sys
import inspect
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union

# Asynchronisme et exécution concurrente
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles

# FastAPI et ses dépendances
from fastapi import FastAPI, File, Request, Query, UploadFile, HTTPException, Form, Response, APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum

# Bases de données et ORM
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Bibliothèques pour le traitement du langage naturel
import torch
from huggingface_hub import HfApi

# Surveillance des ressources
import psutil
import GPUtil

# Modules spécifiques aux fonctionnalités mentionnées
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types import DoclingDocument
from docling_core.types.doc import PictureItem

import semchunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker import BaseChunk, BaseChunker, DocMeta, HierarchicalChunker

from langchain.schema import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFDirectoryLoader

# GLiNER imports
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.data_processing import GLiNERDataset

# Surveillance des performances et des métriques
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

# Autres dépendances
from transformers import AutoTokenizer
import boto3
from loguru import logger
from codecarbon import EmissionsTracker
from pydantic import BaseModel, PositiveInt, Field
from neo4j import GraphDatabase, Transaction
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

load_dotenv()
app = FastAPI(title="Document Processing and GLiNER API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger Setup
logger.add("logs/conversion_{time}.log", rotation="1 day", retention="7 days", level="INFO")
# Metrics
Instrumentator().instrument(app).expose(app)
# Custom metrics
start_http_server(8002)

REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")
MODEL_LOG_COUNT = Counter("model_log_count", "Nombre de modèles enregistrés dans MLflow")
NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Nombre de requêtes envoyées à Neo4j")
NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Nombre de requêtes Neo4j échouées")
NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latence des requêtes Neo4j")

POSTGRES_QUERY_COUNT = Counter("postgres_query_count", "Nombre de requêtes PostgreSQL réussies")
POSTGRES_QUERY_FAILURES = Counter("postgres_query_failures", "Nombre de requêtes PostgreSQL échouées")
POSTGRES_QUERY_LATENCY = Histogram("postgres_query_latency_seconds", "Latence des requêtes PostgreSQL")

DOCUMENT_PROCESSING_SUCCESS = Counter("document_processing_success", "Nombre de documents traités avec succès")
DOCUMENT_PROCESSING_FAILURES = Counter("document_processing_failures", "Nombre d'échecs de traitement de documents")

emissions_tracker = EmissionsTracker(project_name="doc_processing", save_to_file=False, save_to_prometheus=True, prometheus_url="localhost:8002")

# Neo4j setup
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# PostgreSQL setup for MLflow Tracking and Model Registry
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgre_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgre_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgre_db")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
PG_VECTOR_STORE=f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
# Set PostgreSQL as the MLflow tracking and model registry URI
mlflow.set_tracking_uri(DATABASE_URL)
mlflow.set_registry_uri(DATABASE_URL)

# Configure MinIO for artifact storage
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MLFLOW_ARTIFACT_URI = f"s3://{MINIO_ACCESS_KEY}:{MINIO_SECRET_KEY}@{MINIO_URL}/mlflow"

# Set artifact URI separately for artifact storage
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_URL

# PostgreSQL setup for SQLAlchemy
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# S3 (MinIO) setup for artifact storage
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)
input_bucket = 'docs-input'
output_bucket = 'docs-output'
layouts_bucket = 'layouts'

# Ensure the buckets exist
for bucket in [input_bucket, output_bucket, layouts_bucket]:
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Bucket '{bucket}' already exists.")
    except:
        s3_client.create_bucket(Bucket=bucket)
        logger.info(f"Bucket '{bucket}' created.")

# Ensure directories exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# List of available models
AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

def initialize_emissions_tracker():
    """
    Initialisation d'un tracker CodeCarbon avec nettoyage préalable du fichier de verrouillage.
    """
    global emissions_tracker
    lock_file = "/tmp/.codecarbon.lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            logger.info("Fichier de verrouillage CodeCarbon supprimé.")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier de verrouillage CodeCarbon : {e}")

    emissions_tracker = EmissionsTracker(allow_multiple_runs=True)
    logger.info("Tracker CodeCarbon initialisé.")

# Middleware pour collecter les métriques personnalisées
@app.middleware("http")
async def custom_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()  # Incrémentation du compteur de requêtes
    response = await call_next(request)
    latency = time.time() - start_time

    PROCESS_TIME.observe(latency)
    log_system_metrics()  # Log des métriques système

    return response

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

@app.on_event("startup")
async def startup_event():
    initialize_emissions_tracker()
    log_system_metrics()
    try:
        logger.info("Starting application...")
        ensure_vector_index()
        logger.info("Neo4j vector index setup completed successfully.")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise HTTPException(status_code=500, detail="Error during application startup.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application arrêtée.")

# Initialisation du modèle d'embedding et PGVector
embedding_model = OllamaEmbeddings(model="llama3.2")
connection_string = DATABASE_URL
vectorstore = PGVector(
    embeddings=embedding_model,
    collection_name="document_embeddings",
    connection=PG_VECTOR_STORE,
    use_jsonb=True
)
text_splitter = CharacterTextSplitter(chunk_size=1000)

# Initialize Neo4j Graph
neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# GLiNER extractor and transformer
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    glirel_model="jackboyla/glirel-large-v0",
    entity_confidence_threshold=0.3,
    relationship_confidence_threshold=0.3,
)

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False

class RetrievalQuery(BaseModel):
    query: str
    top_k: int = 5

class TextInput(BaseModel):
    text: str = Field(..., example="IBM Watson defeated human champions in the game of Jeopardy!")

class NERInput(BaseModel):
    text: str = Field(..., example="IBM Watson defeated human champions in the game of Jeopardy!")
    ner_model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    labels: Optional[str] = Field(None, example="person, organization, location")
    threshold: Optional[float] = Field(0.5, example=0.5)
    nested_ner: Optional[bool] = Field(False, example=False)

class NEROutput(BaseModel):
    text: str
    entities: List[Dict[str, Union[str, int, float]]] = Field(..., example=[
        {"entity": "organization", "word": "IBM", "start": 0, "end": 3, "score": 0.98}
    ])

class AnnotateInput(BaseModel):
    model: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    labels: str = Field(..., example="person, organization, location")
    threshold: float = Field(0.5, example=0.5)
    prompt: Optional[str] = Field(None, example="Please annotate the following text:")
    sentences: List[str] = Field(..., example=["Google is building a new office in New York."])

class TrainInput(BaseModel):
    ner_model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    custom_model_name: str = Field(..., example="my-custom-model")
    train_data: str  # Path to the training data
    split_ratio: float = Field(0.9, example=0.9)
    learning_rate: float = Field(5e-6, example=5e-6)
    weight_decay: float = Field(0.01, example=0.01)
    batch_size: int = Field(8, example=8)
    epochs: int = Field(1, example=1)
    compile_model: bool = Field(False, example=False)

class EvaluateInput(BaseModel):
    ner_model_name: str = Field(..., example="my-custom-model")

class EvaluateOutput(BaseModel):
    f1_score: float = Field(..., example=0.85)
    results: str = Field(..., example="Entity-wise F1 score: ...")

# Helper functions
def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['entity'] == current['entity'] and (
            next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']
        ):
            current['word'] += ' ' + next_entity['word']
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged

def annotate_text(
    model, text, labels: List[str], threshold: float, nested_ner: bool
) -> Dict:
    labels = [label.strip() for label in labels]
    r = {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": entity.get("score", 0),
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }
    r["entities"] = merge_entities(r["entities"])
    return r

# Device Manager
class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.using_gpu = self.device.type == "cuda"
        logger.info(f"Utilisation de : {self.device}")

    def log_device_stats(self):
        if self.using_gpu:
            for gpu in GPUtil.getGPUs():
                GPU_MEMORY_USAGE.set(gpu.memoryUsed)
                gpu_usage_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                logger.info(f"GPU {gpu.id} - Mémoire utilisée: {gpu.memoryUsed}MB ({gpu_usage_percent:.2f}%)")

        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info()
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory.rss)
        logger.info(f"CPU: {cpu_percent}% | Mémoire: {memory.rss / 1024 / 1024:.2f}MB")

device_manager = DeviceManager()
device = device_manager.device

# Global variables
annotator = None
model_generator = None
# Define ModelManager
class ModelManager:
    def __init__(self):
        self.device_manager = device_manager
        self.tracker_active = False
        self.emissions_tracker = None

    async def process_document(self, doc_converter, doc_path, model_name: str):
        # Ensure any previous MLflow run is ended before starting a new one
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"Processing {model_name}"):
            # Initialize CodeCarbon tracker if none is active
            if not self.tracker_active:
                try:
                    self.emissions_tracker = EmissionsTracker(project_name="doc_processing")
                    self.emissions_tracker.start()
                    self.tracker_active = True
                except Exception as e:
                    logger.warning(f"Unable to start CodeCarbon: {e}")
                    self.emissions_tracker = None

            start_time = time.time()
            result = list(doc_converter.convert_all([doc_path]))[0]
            inference_time = time.time() - start_time

            # Capture emissions if CodeCarbon tracker is active
            emissions = None
            if self.emissions_tracker and self.tracker_active:
                try:
                    emissions = self.emissions_tracker.stop()
                    CARBON_EMISSIONS.set(emissions)
                except Exception as e:
                    logger.warning(f"Error stopping CodeCarbon tracker: {e}")
                finally:
                    self.tracker_active = False  # Reset for next use

            # Log metrics in MLflow
            mlflow.log_metric("inference_time", inference_time)
            if emissions is not None:
                mlflow.log_metric("carbon_emissions", emissions)

            # Log hardware resource usage
            self.device_manager.log_device_stats()
            if self.device_manager.using_gpu:
                gpu_memory_usage = GPUtil.getGPUs()[0].memoryUsed
                mlflow.log_metric("gpu_memory_usage", gpu_memory_usage)

            mlflow.log_metric("cpu_usage", psutil.cpu_percent())
            mlflow.log_metric("memory_usage", psutil.Process().memory_info().rss / (1024 * 1024))  # MB

        return result


    def load_model(self, model_name: str):
        # Chargement du modèle en utilisant la logique existante
        model_path = MODEL_DIR / model_name
        if model_path.exists():
            model = GLiNER.from_pretrained(str(model_path)).to(device_manager.device)
            return model
        elif model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(model_name).to(device_manager.device)
            model.save_pretrained(model_path)
            return model
        else:
            raise HTTPException(status_code=404, detail="Modèle non trouvé.")


model_manager = ModelManager()

class AutoAnnotator:
    def __init__(
        self, model_name: str, device: torch.device, model_manager: ModelManager
    ) -> None:
        self.model_manager = model_manager
        self.model = self.model_manager.load_model(model_name)
        self.annotated_data = []
        self.stat = {
            "total": None,
            "current": -1
        }

    def auto_annotate(
        self, data: List[str], labels: List[str],
        prompt: Optional[str] = None, threshold: float = 0.5, nested_ner: bool = False
    ) -> List[Dict]:
        self.stat["total"] = len(data)
        self.stat["current"] = -1  # Reset current progress
        for text in data:
            self.stat["current"] += 1
            if isinstance(prompt, list):
                prompt_text = random.choice(prompt)
            else:
                prompt_text = prompt
            text_with_prompt = f"{prompt_text}\n{text}" if prompt_text else text

            annotation = annotate_text(self.model, text_with_prompt, labels, threshold, nested_ner)

            if not annotation["entities"]:  # If no entities identified
                annotation = {"text": text, "entities": []}

            self.annotated_data.append(annotation)
        return self.annotated_data

class ModelGenerator:
    def __init__(self) -> None:
        self.previous_path = None
        self.path = None
        self.model = None

    def get_model(self, path):
        if self.path != path:
            self.model = GLiNER.from_pretrained(path, load_tokenizer=True).to(device)
            self.path = path
        return self.model

model_generator = ModelGenerator()

class ModelLoggerService:
    def __init__(self, db_url: str):
        mlflow.set_tracking_uri(db_url)
        self.client = MlflowClient()
        self.hf_api = HfApi()
        self.huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
        self.static_models = {
            "Embedding Model": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(self.huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2")),
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
            model_version = model_info.sha
            model_description = self._fetch_readme(model_id) or "No description available."
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

class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    s3_url = Column(String)

Base.metadata.create_all(bind=engine)

class DocumentLogService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def log_document(self, file_name: str, s3_url: str):
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        with self.session_factory() as session:
            session.add(log)
            session.commit()

class S3Service:
    def __init__(self, client, input_bucket, output_bucket, layouts_bucket):
        self.client = client
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.layouts_bucket = layouts_bucket

    def upload_file(self, file_path: Path, bucket_name: str) -> Optional[str]:
        try:
            self.client.upload_file(str(file_path), bucket_name, file_path.name)
            return f"s3://{bucket_name}/{file_path.name}"
        except Exception as e:
            logger.error(f"Failed to upload file {file_path.name}: {e}")
            return None

class MLFlowService:
    def __init__(self, db_url):
        mlflow.set_tracking_uri(db_url)
        self.client = MlflowClient()

    def log_params(self, params: Dict[str, Any]):
        for param, value in params.items():
            mlflow.log_param(param, value)

    def log_metrics(self, metrics: Dict[str, Any]):
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

class DocumentProcessor:
    def __init__(self, s3_service: S3Service, mlflow_service: MLFlowService, session_factory):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.doc_log_service = DocumentLogService(session_factory)

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

    def export_document(self, result, output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool):
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

    def _export_file(self, result, output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool, doc_filename: str):
        if ExportFormat.json in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "json", export_format="json")
        if ExportFormat.yaml in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "yaml", export_format="yaml")
        if ExportFormat.md in export_formats:
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

s3_service = S3Service(
    client=s3_client,
    input_bucket=input_bucket,
    output_bucket=output_bucket,
    layouts_bucket=layouts_bucket
)
mlflow_service = MLFlowService(db_url=DATABASE_URL)

def ensure_vector_index():
    """
    Ensure that the vector index exists in PostgreSQL.
    """
    # This function can be expanded based on the vectorstore initialization requirements
    pass



@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[ExportFormat] = Query(default=[ExportFormat.json]),
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False
):
    REQUEST_COUNT.inc()
    logger.info(f"Received {len(files)} files for upload")

    doc_processor = DocumentProcessor(s3_service, mlflow_service, SessionLocal)
    converter = doc_processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)
    success_count, partial_success_count, failure_count = 0, 0, 0

    for file in files:
        temp_file = OUTPUT_DIR / file.filename
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        input_s3_url = s3_service.upload_file(temp_file, s3_service.input_bucket)
        doc_processor.doc_log_service.log_document(file.filename, input_s3_url)

        with mlflow.start_run(run_name="Document Conversion"):
            result = await model_manager.process_document(converter, temp_file, model_name="Docling")
            if result:
                counts = doc_processor.export_document(result, OUTPUT_DIR, export_formats, export_figures, export_tables)
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

@app.post("/upload_path/")
async def upload_path(
    file_path: str = Form("/home/pi/Documents/IF-SRV/4pdfs_subset/"),
    export_formats: List[ExportFormat] = Query(default=[ExportFormat.json]),
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False
):
    REQUEST_COUNT.inc()
    logger.info(f"Processing directory: {file_path}")

    input_dir_path = Path(file_path)
    input_file_paths = [
        file for file in input_dir_path.glob('*')
        if file.suffix.lower() in ['.pdf', '.docx']
    ]

    doc_processor = DocumentProcessor(s3_service, mlflow_service, SessionLocal)
    converter = doc_processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)
    success_count, partial_success_count, failure_count = 0, 0, 0

    for doc_path in input_file_paths:
        input_s3_url = s3_service.upload_file(doc_path, s3_service.input_bucket)
        doc_processor.doc_log_service.log_document(doc_path.name, input_s3_url)

        with mlflow.start_run(run_name="Document Conversion"):
            result = await model_manager.process_document(converter, doc_path, model_name="Docling")
            if result:
                counts = doc_processor.export_document(result, OUTPUT_DIR, export_formats, export_figures, export_tables)
                success_count += counts[0]
                partial_success_count += counts[1]
                failure_count += counts[2]

    return {
        "message": "Directory processed and stored successfully",
        "uploaded_to": s3_service.output_bucket,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count
    }

def clean_text(text: str) -> str:
    """Clean up text to remove unwanted characters and normalize whitespace."""
    text = text.replace("\n", " ").strip()
    return re.sub(r'\s+', ' ', text)

def process_document_for_indexing(doc: Document):
    """Process a document to extract nodes and relationships, adding them to Neo4j."""
    try:
        split_docs = text_splitter.split_documents([doc])
        split_docs = [
            Document(page_content=clean_text(chunk.page_content), metadata=chunk.metadata)
            for chunk in split_docs
        ]
        logger.debug(f"Document split into {len(split_docs)} chunks.")

        # Extract graph data
        graph_docs = gliner_extractor.transform_documents(split_docs)

        # Index into Neo4j
        for graph_doc in graph_docs:
            neo4j_graph.write(graph_doc)
            logger.info("Graph document written to Neo4j.")

        # Index embeddings into PGVector
        vectorstore.add_documents(split_docs)
        logger.info("Documents added to PGVector.")

    except Exception as e:
        logger.error(f"Error processing document: {doc.metadata.get('name', 'unknown')} - {e}")

@app.post("/index_documents/")
def index_documents(folder_path: str):
    """
    Index documents from the given folder into Neo4j and PGVector.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}. Execution will use the GPU.")
    else:
        logger.warning("No GPU detected. Execution will fall back to CPU.")

    logger.info(f"Indexing documents from folder: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    total_docs = len(documents)
    logger.info(f"Loaded {total_docs} documents for indexing.")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_document_for_indexing, documents)

    logger.info(f"Successfully indexed {total_docs} documents.")
    return {
        "message": f"{total_docs} documents indexed into Neo4j and PGVector",
        "gpu_used": device.type == "cuda",
        "gpu_name": gpu_name if device.type == "cuda" else "None",
    }

@app.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    """
    Index a single document into Neo4j and PGVector by extracting entities and embeddings.
    """
    REQUEST_COUNT.inc()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}. Execution will use the GPU.")
    else:
        logger.warning("No GPU detected. Execution will fall back to CPU.")

    # Save uploaded file temporarily
    temp_file = OUTPUT_DIR / file.filename
    async with aiofiles.open(temp_file, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Load and process the document
    try:
        loader = PyPDFDirectoryLoader(temp_file.parent)
        documents = loader.load()
        if not documents:
            raise ValueError("No valid documents found in the uploaded file.")
        logger.info(f"Processing document: {file.filename}")

        # Process the document
        process_document_for_indexing(documents[0])
        logger.info(f"Successfully indexed document: {file.filename}.")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        temp_file.unlink()  # Remove temporary file

@app.post("/retrieve_documents/")
async def retrieve_documents(request: RetrievalQuery):
    # Perform the similarity search
    results = vectorstore.similarity_search(request.query, k=request.top_k)

    # Format and return results
    return {"results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]}

@app.post("/search/")
def hybrid_search(query: str = Form(..., description="Requête de recherche hybride")):
    # Perform the similarity search
    results = vectorstore.similarity_search(query, k=5)

    # Format and return results
    return {"results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]}


def log_models():
    """API endpoint to trigger logging of model details."""
    return model_logger_service.log_model_details()

@app.post("/log_queries/")
def log_queries(query: str):
    """API endpoint to trigger logging of queries."""
    return model_logger_service.log_query(query)

# GLiNER Endpoints
documents_router = APIRouter(prefix="/documents", tags=["Documents"])
gliner_router = APIRouter(prefix="/gliner", tags=["GLiNER"])
logging_router = APIRouter(prefix="/logging", tags=["Logging"])
class GLiNERService:
    def __init__(self, model_manager: ModelManager, s3_service: S3Service, mlflow_service: MLFlowService, doc_processor: DocumentProcessor):
        self.model_manager = model_manager
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.doc_processor = doc_processor

    def ner(self, input_data: NERInput) -> NEROutput:
        try:
            model = self.model_manager.load_model(input_data.model_name)
            labels = [label.strip() for label in input_data.labels.split(",")] if input_data.labels else None

            result = annotate_text(
                model, input_data.text, labels, input_data.threshold, input_data.nested_ner
            )
            return NEROutput(**result)
        except Exception as e:
            logger.error(f"Erreur dans l'endpoint NER : {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def annotate(self, input_data: AnnotateInput) -> Dict[str, Any]:
        try:
            # Intégration de Docling pour extraire des données structurées
            if input_data.document_id:
                # Récupérer le document structuré à partir de l'ID
                structured_data = self.doc_processor.get_structured_data(input_data.document_id)
                sentences = [item['text'] for item in structured_data]
            else:
                sentences = input_data.sentences

            labels = [label.strip() for label in input_data.labels.split(",")]
            annotator = AutoAnnotator(
                input_data.model, device=device_manager.device, model_manager=self.model_manager
            )
            annotated_data = annotator.auto_annotate(
                sentences, labels, input_data.prompt, input_data.threshold
            )

            # Sauvegarde et journalisation
            output_filename = f"annotated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = DATA_DIR / output_filename
            with open(output_path, "w") as file:
                json.dump(annotated_data, file)

            s3_url = self.s3_service.upload_file(output_path, self.s3_service.output_bucket)
            self.mlflow_service.log_params({
                "model": input_data.model,
                "labels": input_data.labels,
                "threshold": input_data.threshold,
                "prompt": input_data.prompt,
                "document_id": input_data.document_id
            })
            self.mlflow_service.log_metrics({"annotated_sentences": len(annotated_data)})

            return {"message": f"Annotations sauvegardées sur {s3_url}"}
        except Exception as e:
            logger.error(f"Erreur dans l'endpoint annotate : {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def train(self, train_input: TrainInput) -> Dict[str, Any]:
        try:
            device = device_manager.device
            custom_model_path = MODEL_DIR / train_input.custom_model_name
            custom_model_path.mkdir(parents=True, exist_ok=True)

            model = self.model_manager.load_model(train_input.model_name)

            # Intégration de Docling pour utiliser des données structurées pour l'entraînement
            if train_input.use_docling_data:
                train_data = self.doc_processor.get_training_data_from_docling()
            else:
                train_data_path = DATA_DIR / train_input.train_data
                if not train_data_path.exists():
                    raise HTTPException(status_code=404, detail="Données d'entraînement non trouvées.")
                with open(train_data_path, "r") as f:
                    train_data = json.load(f)

            # Division des données et préparation des datasets
            random.seed(42)
            random.shuffle(train_data)
            split_idx = int(len(train_data) * train_input.split_ratio)
            train_data_split = train_data[:split_idx]
            test_data = train_data[split_idx:]

            # Sauvegarde des données de test
            test_data_path = DATA_DIR / "test.json"
            with open(test_data_path, "w") as f:
                json.dump(test_data, f)

            # Création des datasets GLiNER
            train_dataset = GLiNERDataset(train_data_split, model.config, data_processor=model.data_processor)
            test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
            data_collator = DataCollatorWithPadding(model.config)

            # Arguments d'entraînement
            training_args = TrainingArguments(
                output_dir=str(custom_model_path),
                learning_rate=train_input.learning_rate,
                weight_decay=train_input.weight_decay,
                per_device_train_batch_size=train_input.batch_size,
                per_device_eval_batch_size=train_input.batch_size,
                num_train_epochs=train_input.epochs,
                evaluation_strategy="epoch",
                save_steps=1000,
                save_total_limit=10,
                dataloader_num_workers=4,
                use_cpu=(device.type == 'cpu'),
                report_to="none",
            )

            # Initialisation du Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=model.data_processor.transformer_tokenizer,
                data_collator=data_collator,
            )

            # Entraînement et journalisation
            with mlflow.start_run(run_name="GLiNER Training"):
                self.mlflow_service.log_params({
                    "model_name": train_input.model_name,
                    "custom_model_name": train_input.custom_model_name,
                    "learning_rate": train_input.learning_rate,
                    "weight_decay": train_input.weight_decay,
                    "batch_size": train_input.batch_size,
                    "epochs": train_input.epochs,
                    "split_ratio": train_input.split_ratio,
                    "use_docling_data": train_input.use_docling_data
                })

                trainer.train()

                model.save_pretrained(custom_model_path)
                s3_url = self.s3_service.upload_file(custom_model_path, self.s3_service.output_bucket)

                mlflow.log_artifacts(str(custom_model_path), artifact_path="models")
                self.mlflow_service.log_metrics({"training_completed": 1})

            return {"message": f"Entraînement terminé. Modèle sauvegardé sur {s3_url}"}
        except Exception as e:
            logger.error(f"Erreur dans l'endpoint train : {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def evaluate(self, evaluate_input: EvaluateInput) -> EvaluateOutput:
        try:
            device = device_manager.device
            model = self.model_manager.load_model(evaluate_input.model_name)

            test_data_path = DATA_DIR / 'test.json'
            if not test_data_path.exists():
                raise HTTPException(status_code=404, detail="Données de test non trouvées.")
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)

            test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)

            results, f1_score = model.evaluate(
                test_dataset,
                flat_ner=True,
                threshold=0.5,
                batch_size=8
            )

            with mlflow.start_run(run_name="GLiNER Evaluation"):
                self.mlflow_service.log_metrics({"f1_score": f1_score})
                mlflow.log_param("model_name", evaluate_input.model_name)

            return EvaluateOutput(f1_score=f1_score, results=results)
        except Exception as e:
            logger.error(f"Erreur dans l'endpoint evaluate : {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def zip_and_upload(self, model_name: str) -> Dict[str, Any]:
        try:
            model_path = MODEL_DIR / model_name
            zip_path = model_path.with_suffix('.zip')

            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Répertoire du modèle '{model_name}' non trouvé.")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = Path(root) / file
                        zipf.write(file_path, arcname=file_path.relative_to(model_path))

            s3_url = self.s3_service.upload_file(zip_path, self.s3_service.output_bucket)
            if s3_url:
                zip_path.unlink()
                with mlflow.start_run(run_name="Model Zip and Upload"):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_artifact(str(zip_path), artifact_path="model_zips")
                return {"message": f"Modèle '{model_name}' compressé et téléchargé sur {s3_url}"}
            else:
                raise HTTPException(status_code=500, detail="Échec du téléchargement du fichier zip sur S3.")
        except Exception as e:
            logger.error(f"Erreur dans l'endpoint zip_and_upload : {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Instanciation du GLiNERService
doc_processor = DocumentProcessor(s3_service, mlflow_service, SessionLocal)
gliner_service = GLiNERService(model_manager, s3_service, mlflow_service, doc_processor)

# GLiNER Router Endpoints
@gliner_router.post("/ner/", response_model=NEROutput)
def ner_endpoint(input_data: NERInput):
    return gliner_service.ner(input_data)

@gliner_router.post("/annotate/")
def annotate_endpoint(input_data: AnnotateInput):
    return gliner_service.annotate(input_data)

@gliner_router.post("/train/")
def train_endpoint(train_input: TrainInput):
    return gliner_service.train(train_input)

@gliner_router.post("/evaluate/", response_model=EvaluateOutput)
def evaluate_endpoint(evaluate_input: EvaluateInput):
    return gliner_service.evaluate(evaluate_input)

@gliner_router.post("/zip_and_upload/")
def zip_and_upload_endpoint(model_name: str = Form(..., description="Nom du modèle à compresser et uploader")):
    return gliner_service.zip_and_upload(model_name)

# Logging Router Endpoints
@logging_router.post("/log_models/")
def log_models():
    return model_logger_service.log_model_details()

@logging_router.post("/log_queries/")
def log_queries(query: str = Form(..., description="Requête à journaliser")):
    return model_logger_service.log_query(query)

# Inclusion des routers dans l'application principale
app.include_router(documents_router)
app.include_router(gliner_router)
app.include_router(logging_router)

# L'endpoint pour les métriques reste inchangé
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)