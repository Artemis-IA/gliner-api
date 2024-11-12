import os, time, json, yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import aiofiles
from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch
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
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.document_loaders import PyPDFDirectoryLoader
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import boto3, re
from loguru import logger
from codecarbon import EmissionsTracker
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv
import mlflow

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

# Logger Setup
logger.add("logs/conversion_{time}.log", rotation="1 day", retention="7 days", level="INFO")

# Neo4j and MLflow setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
mlflow.set_tracking_uri("http://localhost:5002")

# PostgreSQL setup
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# S3 (MinIO) setup
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_access_key_id='minio',
    aws_secret_access_key='minio123'

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

# Initialisation du modèle et de l'embedding
ollama_emb = OllamaEmbeddings(model="llama3.2")
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
        self.emissions_tracker = EmissionsTracker(project_name="doc_processing")

    async def process_document(self, doc_converter, doc_path):
        # if not self.tracker_active:
            # self.emissions_tracker.start()
            # self.tracker_active = True
        # start_time = time.time()

        result = list(doc_converter.convert_all([doc_path]))[0]
        emissions = self.emissions_tracker.stop()
        self.tracker_active = False
        # CARBON_EMISSIONS.set(emissions)

        # PROCESS_TIME.observe(time.time() - start_time)
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
        result = await model_manager.process_document(doc_converter, doc_path)
        
        if result:
            results.append(result)
            export_results = export_documents([result], OUTPUT_DIR, export_formats, export_figures, export_tables)

    logger.info(f"All documents processed. Total successful: {len(results)}")
    return {"message": "Documents processed successfully", "uploaded_to": output_bucket}


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
        logger.info(f"Uploaded original file '{doc_path.name}' to input bucket: {input_s3_url}")
        
        logger.info(f"Converting document: {doc_path}")
        result = await model_manager.process_document(doc_converter, doc_path)
        
        if result:
            results.append(result)
            export_results = export_documents([result], OUTPUT_DIR, export_formats, export_figures, export_tables)

    logger.info(f"Directory {file_path} processed. Total successful: {len(results)}")
    return {"message": "Directory processed successfully", "uploaded_to": output_bucket}


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
