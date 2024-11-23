import os
import time
import json
import yaml
import sys
import inspect
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator

# Asynchronisme et exécution concurrente
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles

# FastAPI et ses dépendances
from fastapi import FastAPI, File, Request, Query, UploadFile, HTTPException, Form, Response
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

from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_postgres import PGVector
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Surveillance des performances et des métriques
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

# Autres dépendances
from transformers import AutoTokenizer
import boto3
import re
from loguru import logger
from codecarbon import EmissionsTracker
from pydantic import BaseModel, PositiveInt
from neo4j import GraphDatabase, Transaction
from py2neo import Graph, NodeMatcher, Relationship
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

load_dotenv()
app = FastAPI(title="Document Processing API", version="1.0.0")
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
NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Number of requests sent to Neo4j")
NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Number of failed Neo4j requests")
NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latency of Neo4j requests")

POSTGRES_QUERY_COUNT = Counter("postgres_query_count", "Number of successful PostgreSQL queries")
POSTGRES_QUERY_FAILURES = Counter("postgres_query_failures", "Number of failed PostgreSQL queries")
POSTGRES_QUERY_LATENCY = Histogram("postgres_query_latency_seconds", "Latency of PostgreSQL queries")

DOCUMENT_PROCESSING_SUCCESS = Counter("document_processing_success", "Number of successfully processed documents")
DOCUMENT_PROCESSING_FAILURES = Counter("document_processing_failures", "Number of failed document processing attempts")

emissions_tracker = EmissionsTracker(project_name="doc_processing", save_to_file=False, save_to_prometheus=True, prometheus_url="localhost:8002")

# Neo4j setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# PostgreSQL setup for MLflow Tracking and Model Registry
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgre_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgre_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgre_db")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"

# Set PostgreSQL as the MLflow tracking and model registry URI
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_API_URL", "http://localhost:9000")
mlflow.set_tracking_uri(DATABASE_URL)
mlflow.set_registry_uri(DATABASE_URL)

# Configure MinIO for artifact storage
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MLFLOW_ARTIFACT_URI = f"s3://{MINIO_ACCESS_KEY}:{MINIO_SECRET_KEY}@{MINIO_URL}/mlflow"

# Set artifact URI separately for artifact storage
mlflow.set_tracking_uri(DATABASE_URL)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_URL
mlflow.set_tracking_uri(DATABASE_URL)

# PostgreSQL setup for SQLAlchemy
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


# FastAPI App
app = FastAPI(title="Document Processing API", version="2.0.0")
logger.add("logs/conversion_{time}.log", rotation="1 day", retention="7 days", level="INFO")

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
        ensure_vector_index(driver)
        logger.info("Neo4j vector index setup completed successfully.")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise HTTPException(status_code=500, detail="Error during application startup.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application arrêtée.")

# Initialisation du modèle d'embedding Ollama et PGVector
ollama_emb = OllamaEmbeddings(model="llama3.2")
connection_string = "postgresql+psycopg://postgre_user:postgre_password@localhost/postgre_db"
vectorstore = PGVector(
    embeddings=ollama_emb,
    collection_name="document_embeddings",
    connection=connection_string,
    use_jsonb=True
)
text_splitter = CharacterTextSplitter(chunk_size=1000)

# Initialize Neo4jVector at app startup
def ensure_vector_index(driver):
    """
    Ensure that the vector index exists in Neo4j.
    """
    query = """
    CREATE INDEX vector_index IF NOT EXISTS
    FOR (n:Vector)
    ON (n.vector)
    """
    with driver.session() as session:
        try:
            session.run(query)
            print("Index 'vector_index' ensured.")
        except Exception as e:
            print(f"Error ensuring index: {e}")
            raise e

# GLiNER extractor and transformer
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="E3-JSI/gliner-multi-pii-domains-v1"
)
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    glirel_model="jackboyla/glirel-large-v0",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False 

class RetrievalQuery(BaseModel):
    query: str
    top_k: int = 5



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

# Device and Model Manager
class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.using_gpu = self.device.type == "cuda"
        logger.info(f"Utilisation de : {self.device}")
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

class ModelManager:
    def __init__(self):
        self.device_manager = DeviceManager()
        self.tracker_active = False
        self.emissions_tracker = None  # Initialize the tracker as None

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


model_manager = ModelManager()

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


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

def count_tokens(text: list[str] | None, tokenizer):
    if text is None:
        return 0
    elif isinstance(text, list):
        total = sum(count_tokens(t, tokenizer) for t in text)
        return total
    return len(tokenizer.tokenize(text, max_length=None))

def make_splitter(tokenizer, chunk_size):
    return semchunk.chunkerify(tokenizer, chunk_size)

def doc_chunk_length(doc_chunk: DocChunk, tokenizer):
    text_length = count_tokens(doc_chunk.text, tokenizer)
    headings_length = count_tokens(doc_chunk.meta.headings, tokenizer)
    captions_length = count_tokens(doc_chunk.meta.captions, tokenizer)
    total = text_length + headings_length + captions_length
    return {"total": total, "text": text_length, "other": total - text_length}

def make_chunk_from_doc_items(
    doc_chunk: DocChunk, window_text: str, window_start: int, window_end: int
) -> DocChunk:
    meta = DocMeta(
        doc_items=doc_chunk.meta.doc_items[window_start:window_end + 1],
        headings=doc_chunk.meta.headings,
        captions=doc_chunk.meta.captions,
    )
    new_chunk = DocChunk(text=window_text, meta=meta)
    return new_chunk

def merge_text(t1: str, t2: str) -> str:
    if t1 == "":
        return t2
    elif t2 == "":
        return t1
    else:
        return t1 + "\n" + t2

def split_by_doc_items(doc_chunk: DocChunk, tokenizer, chunk_size: int) -> List[DocChunk]:
    if doc_chunk.meta.doc_items is None or len(doc_chunk.meta.doc_items) <= 1:
        return [doc_chunk]
    length = doc_chunk_length(doc_chunk, tokenizer)
    if length["total"] <= chunk_size:
        return [doc_chunk]
    else:
        chunks = []
        window_start = 0
        window_end = 0
        window_text = ""
        window_text_length = 0
        other_length = length["other"]
        l = len(doc_chunk.meta.doc_items)
        while window_end < l:
            doc_item = doc_chunk.meta.doc_items[window_end]
            text = doc_item.text
            text_length = count_tokens(text, tokenizer)
            if (
                text_length + window_text_length + other_length < chunk_size
                and window_end < l - 1
            ):
                window_end += 1
                window_text_length += text_length
                window_text = merge_text(window_text, text)
            elif text_length + window_text_length + other_length < chunk_size:
                window_text = merge_text(window_text, text)
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end
                )
                chunks.append(new_chunk)
                window_end = l
            elif window_start == window_end:
                window_text = merge_text(window_text, text)
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end
                )
                chunks.append(new_chunk)
                window_start = window_end + 1
                window_end = window_start
                window_text = ""
                window_text_length = 0
            else:
                new_chunk = make_chunk_from_doc_items(
                    doc_chunk, window_text, window_start, window_end - 1
                )
                chunks.append(new_chunk)
                window_start = window_end
                window_text = ""
                window_text_length = 0

        return chunks

def split_using_plain_text(
    doc_chunk: DocChunk,
    tokenizer,
    plain_text_splitter,
    chunk_size: int,
) -> List[DocChunk]:
    lengths = doc_chunk_length(doc_chunk, tokenizer)
    if lengths["total"] <= chunk_size:
        return [doc_chunk]
    else:
        available_length = chunk_size - lengths["other"]
        if available_length <= 0:
            raise ValueError(
                "Headers and captions for this chunk are longer than the total amount of size for the chunk. This is not supported now."
            )
        text = doc_chunk.text
        segments = plain_text_splitter.chunk(text)
        chunks = []
        for s in segments:
            new_chunk = DocChunk(text=s, meta=doc_chunk.meta)
            chunks.append(new_chunk)
        return chunks

def merge_chunks_with_matching_metadata(chunks, tokenizer, chunk_size):
    output_chunks = []
    window_start = 0
    window_end = 0
    l = len(chunks)
    while window_end < l:
        chunk = chunks[window_end]
        lengths = doc_chunk_length(chunk, tokenizer)
        headings_and_captions = (chunk.meta.headings, chunk.meta.captions)
        if window_start == window_end:
            current_headings_and_captions = headings_and_captions
            window_text = chunk.text
            window_other_length = lengths["other"]
            window_text_length = lengths["text"]
            window_items = chunk.meta.doc_items
            window_end += 1
            first_chunk_of_window = chunk
        elif (
            headings_and_captions == current_headings_and_captions
            and window_text_length + window_other_length + lengths["text"] <= chunk_size
        ):
            window_text = merge_text(window_text, chunk.text)
            window_text_length += lengths["text"]
            window_items = window_items + chunk.meta.doc_items
            window_end += 1
        else:
            if window_start + 1 == window_end:
                output_chunks.append(first_chunk_of_window)
            else:
                new_meta = DocMeta(
                    doc_items=window_items,
                    headings=headings_and_captions[0],
                    captions=headings_and_captions[1],
                )
                new_chunk = DocChunk(text=window_text, meta=new_meta)
                output_chunks.append(new_chunk)
            window_start = window_end

    return output_chunks

def merge_chunks_with_mismatching_metadata(chunks, *_):
    return chunks

def merge_chunks(chunks, tokenizer, chunk_size):
    initial_merged_chunks = merge_chunks_with_matching_metadata(
        chunks, tokenizer, chunk_size
    )
    final_merged_chunks = merge_chunks_with_mismatching_metadata(
        initial_merged_chunks, tokenizer, chunk_size
    )
    return final_merged_chunks

def adjust_chunks_for_fixed_size(doc, original_chunks, tokenizer, splitter, chunk_size):
    chunks_after_splitting_by_items = []
    for chunk in original_chunks:
        chunk_split_by_doc_items = split_by_doc_items(chunk, tokenizer, chunk_size)
        chunks_after_splitting_by_items.extend(chunk_split_by_doc_items)
    chunks_after_splitting_recursively = []
    for chunk in chunks_after_splitting_by_items:
        chunk_split_recursively = split_using_plain_text(
            chunk, tokenizer, splitter, chunk_size
        )
        chunks_after_splitting_recursively.extend(chunk_split_recursively)
    chunks_after_merging = merge_chunks(
        chunks_after_splitting_recursively, tokenizer, chunk_size
    )
    return chunks_after_merging

class MaxTokenLimitingChunkerWithMerging(BaseChunker):
    inner_chunker: BaseChunker = HierarchicalChunker()
    max_tokens: PositiveInt = 512
    embedding_model_id: str

    def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]:
        preliminary_chunks = self.inner_chunker.chunk(dl_doc=dl_doc, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_id)
        splitter = make_splitter(tokenizer, self.max_tokens)
        output_chunks = adjust_chunks_for_fixed_size(
            dl_doc, preliminary_chunks, tokenizer, splitter, self.max_tokens
        )
        return iter(output_chunks)

s3_service = S3Service(
    client=s3_client,
    input_bucket=input_bucket,
    output_bucket=output_bucket,
    layouts_bucket=layouts_bucket
)    
mlflow_service = MLFlowService(db_url=DATABASE_URL)

class DocumentProcessingPipeline: 
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

    async def preprocess_document(self, file_path: Path, converter: DocumentConverter):
        """Effectuer le prétraitement d'un document via Docling."""
        logger.info(f"Prétraitement du document : {file_path.name}")
        try:
            result = list(converter.convert_all([file_path]))[0]
            return result
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement du document {file_path.name}: {e}")
            return None

    def process_graph_extraction(self, document: DoclingDocument):
        """Extraire des entités et relations avec GLiNER et Graph Transformer."""
        try:
            logger.info(f"Extraction des entités et relations pour le document : {document.meta.source}")
            split_docs = text_splitter.split_documents([document])
            split_docs = [
                Document(page_content=clean_text(chunk.page_content), metadata=chunk.metadata)
                for chunk in split_docs
            ]

            # Utilisation de GLiNER et Graph Transformer
            graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
            doc_links = [gliner_extractor.extract_one(chunk) for chunk in split_docs]

            with driver.session() as session:
                with session.begin_transaction() as tx:
                    for graph_doc, links in zip(graph_docs, doc_links):
                        self._add_nodes_to_neo4j(tx, graph_doc.nodes)
                        self._add_edges_to_neo4j(tx, graph_doc.edges)
                        self._add_links_to_neo4j(tx, links)
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des entités et relations : {e}")

    @staticmethod
    def _add_nodes_to_neo4j(tx, nodes):
        if not nodes:
            return
        for node in nodes:
            tx.run(
                """
                MERGE (e:Entity {id: $id, name: $name, type: $type})
                ON CREATE SET e.created_at = timestamp()
                """,
                {
                    "id": node.id,
                    "name": node.properties.get("name", ""),
                    "type": node.type,
                },
            )
            logger.info(f"Indexé Node: {node.id}, Type: {node.type}")

    @staticmethod
    def _add_edges_to_neo4j(tx, edges):
        if not edges:
            return
        for edge in edges:
            tx.run(
                """
                MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                MERGE (source)-[r:$type]->(target)
                """,
                {
                    "source_id": edge.source.id,
                    "target_id": edge.target.id,
                    "type": edge.type,
                },
            )

    @staticmethod
    def _add_links_to_neo4j(tx, links):
        if not links:
            return
        for link in links:
            tx.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.created_at = timestamp()
                RETURN e
                """,
                {"name": link.tag},
            )

mlflow_service = MLFlowService(db_url=DATABASE_URL)
doc_processor = DocumentProcessingPipeline(s3_service, mlflow_service, SessionLocal)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/process_document/")
async def process_document_endpoint(
    file: UploadFile = File(...),
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False,
):
    """
    Endpoint to preprocess and index a single document into Neo4j.
    """
    logger.info(f"Processing file: {file.filename}")
    converter = doc_processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)

    # Save the uploaded file temporarily
    temp_file = OUTPUT_DIR / file.filename
    async with aiofiles.open(temp_file, "wb") as out_file:
        await out_file.write(await file.read())

    try:
        # Preprocess the document with Docling
        result = await doc_processor.preprocess_document(temp_file, converter)
        if not result:
            raise HTTPException(status_code=500, detail="Error preprocessing document.")

        # Extract entities and relationships using GLiNER
        doc_processor.process_graph_extraction(result.document)
        logger.info(f"Successfully processed and indexed document: {file.filename}")

        return {"message": f"Document {file.filename} successfully processed and indexed into Neo4j."}
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")
    finally:
        temp_file.unlink()


@app.post("/process_directory/")
async def process_directory_endpoint(
    folder_path: str,
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False,
):
    """
    Endpoint to process all documents in a folder.
    """
    logger.info(f"Processing directory: {folder_path}")
    converter = doc_processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)

    input_dir = Path(folder_path)
    input_files = [file for file in input_dir.glob("*.pdf")]

    if not input_files:
        raise HTTPException(status_code=400, detail="No valid files found in the directory.")

    success_count = 0
    for file in input_files:
        result = await doc_processor.preprocess_document(file, converter)
        if result:
            doc_processor.process_graph_extraction(result.document)
            success_count += 1

    logger.info(f"Successfully processed {success_count}/{len(input_files)} documents from the directory.")
    return {"message": f"{success_count}/{len(input_files)} documents successfully processed and indexed into Neo4j."}


@app.post("/index_documents/")
def index_documents(folder_path: str):
    """
    Directly index documents from the folder into Neo4j.
    """
    logger.info(f"Indexing documents from folder: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    total_docs = len(documents)

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(doc_processor.process_graph_extraction, documents)

    logger.info(f"Successfully indexed {total_docs} documents into Neo4j.")
    return {"message": f"{total_docs} documents indexed into Neo4j"}


@app.post("/verify_index/")
def verify_index():
    """
    Verify the state of the Neo4j database.
    """
    try:
        with driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN COUNT(n) AS total_nodes").single()["total_nodes"]
            total_relationships = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS total_relationships").single()["total_relationships"]

        logger.info(f"Neo4j contains {total_nodes} nodes and {total_relationships} relationships.")
        return {"total_nodes": total_nodes, "total_relationships": total_relationships}
    except Exception as e:
        logger.error(f"Error verifying Neo4j index: {e}")
        raise HTTPException(status_code=500, detail="Error verifying index.")


@app.post("/retrieve_documents/")
async def retrieve_documents(query: str, top_k: int = Query(default=5)):
    """
    Retrieve documents from the vector store using a query.
    """
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return {"results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving documents.")


@app.get("/metrics")
async def metrics():
    """
    Endpoint to expose metrics for Prometheus.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)