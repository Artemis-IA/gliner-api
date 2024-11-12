import os, time, json, yaml, sys, inspect
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator
import aiofiles
from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch
from huggingface_hub import HfApi, ModelInfo
import psutil, GPUtil
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.document_loaders import PyPDFDirectoryLoader
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from transformers import AutoTokenizer
import boto3, re
from loguru import logger
from codecarbon import EmissionsTracker
from pydantic import BaseModel, PositiveInt
from neo4j import GraphDatabase
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

# Charger les variables d'environnement
load_dotenv()

# Prometheus Metrics
start_http_server(8001)
REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")
MODEL_LOG_COUNT = Counter("model_log_count", "Nombre de modèles enregistrés dans MLflow")

# Logger Setup
logger.add("logs/conversion_{time}.log", rotation="1 day", retention="7 days", level="INFO")

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

# Initialisation du modèle d'embedding Ollama et PGVector
ollama_emb = OllamaEmbeddings(model="llama3.2")
connection_string = "postgresql+psycopg://postgre_user:postgre_password@localhost/postgre_db"
vectorstore = PGVector(
    embeddings=ollama_emb,
    collection_name="document_embeddings",
    connection=connection_string,
    use_jsonb=True
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

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
# Configuration FastAPI
app = FastAPI(title="Document Processing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrievalQuery(BaseModel):
    query: str
    top_k: int = 5  # Number of results to return

def log_dynamic_model_details():
    """
    Logs model details including real Description, Tags, and Versions into MLflow and records them in the MLflow model registry.
    """
    huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub/")
    client = MlflowClient()
    hf_api = HfApi()  # Initialize the Hugging Face API client

    # Define model identifiers instead of file paths
    static_models = {
        "Ollama Embedding Model": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(huggingface_cache, "models--sentence-transformers--all-MiniLM-L6-v2")), 
        "GLiNER Extractor Model": ("E3-JSI/gliner-multi-pii-domains-v1", os.path.join(huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1")),
        "Gliner Transformer Model": ("knowledgator/gliner-multitask-large-v0.5", os.path.join(huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5")),
        "Tokenizer Model": ("microsoft/deberta-v3-large", os.path.join(huggingface_cache, "models--microsoft--deberta-v3-large"))
    }

    # Cache paths for local models
    models_cache_paths = {
        "DeBERTa Model": os.path.join(huggingface_cache, "models--microsoft--deberta-v3-large"),
        "GLiNER Multi-PII Model": os.path.join(huggingface_cache, "models--E3-JSI--gliner-multi-pii-domains-v1"),
        "Gliner Multitask Large Model": os.path.join(huggingface_cache, "models--knowledgator--gliner-multitask-large-v0.5"),
        "GLiREL Large Model": os.path.join(huggingface_cache, "models--jackboyla--glirel-large-v0")
    }

    # Start the MLflow run
    with mlflow.start_run(run_name="Suivi Automatique des Modèles") as run:
        run_id = run.info.run_id  # Capture the run ID

        # Log identifiers and metadata for Hugging Face models
        for model_name, (model_id, model_file_path) in static_models.items():
            try:
                # Check if the model is already registered
                registered_models = [rm.name for rm in client.search_registered_models()]
                if model_name not in registered_models:
                    client.create_registered_model(model_name)
                    logger.info(f"Modèle {model_name} enregistré dans le registre de modèles")
                else:
                    logger.info(f"Modèle {model_name} déjà enregistré, passage à l'étape suivante")

                # Fetch model metadata from Hugging Face API
                model_info = hf_api.model_info(model_id)
                model_description = model_info.cardData.get('model_index', [{}])[0].get('description', 'No description available.')
                model_tags = model_info.tags
                model_version = model_info.sha

                # Log Description, Tags, and Version
                mlflow.set_tag(f"{model_name}_description", model_description)
                for tag in model_tags:
                    mlflow.set_tag(f"{model_name}_tag_{tag}", True)
                mlflow.log_param(f"{model_name}_version", model_version)

                # Log the artifact in the current run's artifact directory
                if os.path.exists(model_file_path):
                    artifact_path = f"artifacts/{model_name}"
                    mlflow.log_artifact(model_file_path, artifact_path=artifact_path)
                    
                # Register the model version with a valid source path and run_id
                client.create_model_version(
                    name=model_name,
                    source=f"{mlflow.get_artifact_uri()}/{artifact_path}",
                    run_id=run_id,
                )
                mlflow.log_param(f"{model_name}_identifier", model_id)
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement du modèle {model_name} : {e}")

        # Log cache paths for local file-based models
        for model_name, path in models_cache_paths.items():
            if os.path.exists(path):
                mlflow.log_param(f"{model_name}_cache_path", path)
                logger.info(f"Chemin de cache pour {model_name} : {path}")
            else:
                mlflow.log_param(f"{model_name}_cache_path", "Non trouvé")
                logger.warning(f"Chemin de cache non trouvé pour {model_name}")

# Call the updated function
log_dynamic_model_details()


# Retrieval route
@app.post("/retrieve_documents/")
async def retrieve_documents(request: RetrievalQuery):
    # Create embeddings instance and vectorstore instance
    ollama_emb = OllamaEmbeddings(model="llama3.2")
    connection_string = "postgresql+psycopg://postgre_user:postgre_password@localhost/postgre_db"
    vectorstore = PGVector(
        embeddings=ollama_emb,
        collection_name="document_embeddings",
        connection=connection_string,
        use_jsonb=True
    )
    
    # Perform the similarity search
    results = vectorstore.similarity_search(request.query, k=request.top_k)
    
    # Format and return results
    return {"results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]}

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

from codecarbon import EmissionsTracker
import mlflow


class ModelManager:
    def __init__(self):
        self.device_manager = DeviceManager()
        self.tracker_active = False
        self.emissions_tracker = None  # Initialiser le tracker en tant que None

    async def process_document(self, doc_converter, doc_path, model_name: str):
        # Démarre une exécution MLflow
        with mlflow.start_run(run_name=f"Processing {model_name}"):
            # Initialisation de CodeCarbon si aucune instance n'est active
            if not self.tracker_active:
                try:
                    self.emissions_tracker = EmissionsTracker(project_name="doc_processing")
                    self.emissions_tracker.start()
                    self.tracker_active = True
                except Exception as e:
                    logger.warning(f"Impossible de démarrer CodeCarbon : {e}")
                    self.emissions_tracker = None

            start_time = time.time()
            result = list(doc_converter.convert_all([doc_path]))[0]
            inference_time = time.time() - start_time

            # Si CodeCarbon est actif, récupérer les émissions
            emissions = None
            if self.emissions_tracker and self.tracker_active:
                try:
                    emissions = self.emissions_tracker.stop()
                    CARBON_EMISSIONS.set(emissions)
                except Exception as e:
                    logger.warning(f"Erreur lors de l'arrêt de CodeCarbon : {e}")
                finally:
                    self.tracker_active = False  # Réinitialise pour la prochaine utilisation

            # Enregistrer les métriques dans MLflow
            mlflow.log_metric("inference_time", inference_time)
            if emissions is not None:
                mlflow.log_metric("carbon_emissions", emissions)

            # Journalisation des ressources matérielles
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

def upload_to_s3(file_name, bucket):
    s3_url = None
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
        s3_url = f"s3://{bucket}/{os.path.basename(file_name)}"
    except:
        logger.error(f"Failed to upload {file_name}")
    return s3_url

def log_to_postgres(file_name, s3_url):
    with SessionLocal() as session:
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        session.add(log)
        session.commit()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False 


def create_document_converter(use_ocr, export_figures, export_tables, enrich_figures):
    pipeline_options = CustomPdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = export_tables
    pipeline_options.generate_picture_images = export_figures
    if enrich_figures:
        pipeline_options.do_picture_classifier = enrich_figures
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
        }
    )

def export_documents(conv_results: List[ConversionResult], output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool) -> Tuple[int, int, int]:
    success_count = 0
    partial_success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1

            # Export des résultats dans les formats sélectionnés
            if ExportFormat.json in export_formats:
                json_path = output_dir / f"{doc_filename}.json"
                with json_path.open("w", encoding='utf-8') as json_file:
                    json.dump(conv_res.document.export_to_dict(), json_file, ensure_ascii=False, indent=2)
                upload_to_s3(json_path, output_bucket)

            if ExportFormat.yaml in export_formats:
                yaml_path = output_dir / f"{doc_filename}.yaml"
                with yaml_path.open("w", encoding='utf-8') as yaml_file:
                    yaml.dump(conv_res.document.export_to_dict(), yaml_file, allow_unicode=True, default_flow_style=False)
                upload_to_s3(yaml_path, output_bucket)

            if ExportFormat.md in export_formats:
                md_path = output_dir / f"{doc_filename}.md"
                with md_path.open("w", encoding='utf-8') as md_file:
                    md_file.write(conv_res.document.export_to_markdown())
                upload_to_s3(md_path, output_bucket)

            # Export des figures (images)
            if export_figures:
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                for idx, element in enumerate(conv_res.document.iterate_items()):
                    if isinstance(element, PictureItem):
                        figure_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                        element.image.pil_image.save(figure_path, format="PNG")
                        upload_to_s3(figure_path, layouts_bucket)

            # Export des tableaux
            if export_tables:
                tables_dir = output_dir / "tables"
                tables_dir.mkdir(exist_ok=True)
                for table_idx, table in enumerate(conv_res.document.tables):
                    csv_path = tables_dir / f"{doc_filename}_table_{table_idx + 1}.csv"
                    table.export_to_dataframe().to_csv(csv_path, index=False, encoding='utf-8')
                    upload_to_s3(csv_path, layouts_bucket)

                    html_path = tables_dir / f"{doc_filename}_table_{table_idx + 1}.html"
                    with html_path.open("w", encoding='utf-8') as html_file:
                        html_file.write(table.export_to_html())
                    upload_to_s3(html_path, layouts_bucket)

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
        else:
            failure_count += 1

    return success_count, partial_success_count, failure_count

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

    doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)
    input_file_paths = []
    
    # Step 1: Upload original files to input bucket
    for file in files:
        temp_file = OUTPUT_DIR / file.filename
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        input_file_paths.append(temp_file)
        input_s3_url = upload_to_s3(temp_file, input_bucket)
        logger.info(f"Uploaded original file '{temp_file.name}' to input bucket: {input_s3_url}")

    # Step 2: Process files and save the converted files to output bucket
    results = []
    for doc_path in input_file_paths:
        logger.info(f"Converting document: {doc_path}")
        with mlflow.start_run(run_name="Document Conversion"):
            mlflow.log_param("file_name", doc_path.name)
            mlflow.log_param("conversion_type", "DocConversion")
            mlflow.log_param("use_ocr", use_ocr)
            # mlflow.log_param("export_figures", export_figures)
            # mlflow.log_param("export_tables", export_tables)
            # mlflow.log_param("enrich_figures", enrich_figures)
            # mlflow.log_param("export_formats", export_formats)
            # mlflow.log_param("input_bucket", input_bucket)
            # mlflow.log_param("output_bucket", output_bucket)
            # mlflow.log_param("layouts_bucket", layouts_bucket)
            # mlflow.log_param("embedding_model_id", EMBED_MODEL_ID)
            # mlflow.log_param("max_tokens", 512)
            # mlflow.log_param("model_manager", model_manager)
            # mlflow.log_param("vectorstore", vectorstore)
            # mlflow.log_param("text_splitter", text_splitter)
            # mlflow.log_param("graph_transformer", graph_transformer)
            # mlflow.log_param("gliner_extractor", gliner_extractor)

        result = await model_manager.process_document(doc_converter, doc_path, model_name="Docling")
        
        if result and result.status == ConversionStatus.SUCCESS:
            results.append(result)
            export_results = export_documents([result], OUTPUT_DIR, export_formats, export_figures, export_tables)

            # Chunk the document, generate embeddings, and store in PGVector
            with mlflow.start_run(run_name="Embedding Generation"):
                chunker = MaxTokenLimitingChunkerWithMerging(max_tokens=512, embedding_model_id=EMBED_MODEL_ID)
                chunks = list(chunker.chunk(dl_doc=result.document))

                mlflow.log_metric("chunk_count", len(chunks))
                documents = [Document(page_content=chunk.text, metadata={"file_name": doc_path.name}) for chunk in chunks]
                vectorstore.add_documents(documents=documents)

                mlflow.log_param("embedding_model", "OllamaEmbeddings")
                mlflow.log_metric("embedding_chunk_count", len(documents))

        mlflow.end_run()

    return {"message": "Documents processed and stored successfully", "uploaded_to": output_bucket}


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

    doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)
    results = []
    
    for doc_path in input_file_paths:
        input_s3_url = upload_to_s3(doc_path, input_bucket)
        result = await model_manager.process_document(doc_converter, doc_path)
        
        if result and result.status == ConversionStatus.SUCCESS:
            results.append(result)
            export_results = export_documents([result], OUTPUT_DIR, export_formats, export_figures, export_tables)

            # Chunk the document, generate embeddings, and store in PGVector
            chunker = MaxTokenLimitingChunkerWithMerging(max_tokens=512, embedding_model_id=EMBED_MODEL_ID)
            chunks = list(chunker.chunk(dl_doc=result.document))

            documents = [Document(page_content=chunk.text, metadata={"file_name": doc_path.name}) for chunk in chunks]
            vectorstore.add_documents(documents=documents)

    return {"message": "Directory processed and stored successfully", "uploaded_to": output_bucket}

@app.post("/index_documents/")
def index_documents(folder_path: str):
    logger.info(f"Indexing documents from folder: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    total_docs = len(documents)
    logger.info(f"Loaded {total_docs} documents for indexing")

    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        with driver.session() as session:
            for graph_doc in graph_docs:
                for node in graph_doc.nodes:
                    logger.debug(f"Indexing entity: {node.id} of type {node.type}")
                    session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})

    logger.info(f"All documents indexed successfully from folder: {folder_path}")
    return {"message": "Documents indexed in Neo4j"}

@app.post("/search/")
def hybrid_search(query: str):
    store = Neo4jVector.from_existing_index(
        ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid",
    )
    retriever = store.as_retriever()
    results = retriever.invoke(query)
    return {"results": results}

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
