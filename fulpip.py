import os, time, json, yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import aiofiles
import pandas as pd
from enum import Enum
import torch

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem, TableItem
from docling_core.transforms.chunker import HierarchicalChunker

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor, LinkExtractorTransformer
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from loguru import logger
import mlflow
import psutil, GPUtil
from codecarbon import EmissionsTracker
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3, re
from botocore.exceptions import ClientError
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dotenv import load_dotenv
load_dotenv()

# Configuration des métriques Prometheus
start_http_server(8001)
REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")


# Configuration Loguru
logger.add(
    "logs/conversion_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,
    diagnose=True
)


# Setup Neo4j connection
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)
mlflow.set_tracking_uri("http://localhost:5002")

# Initialize embeddings and model using Ollama
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# Setup PostgreSQL connection
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define a logging model for PostgreSQL
class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    s3_url = Column(String)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# Configure S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_access_key_id='minio',
    aws_secret_access_key='minio123'

)
bucket_name = 'pdfs'
# Ensure bucket exists or create it if it doesn't
try:
    s3_client.head_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' already exists.")
except ClientError:
    s3_client.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

# GLiNER configuration for graph transformation
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

# Function to upload a file to S3
def upload_to_s3(file_name, bucket):
    """Upload a file to an S3 bucket with progress tracking."""
    s3_url = None
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
        s3_url = f"s3://{bucket}/{os.path.basename(file_name)}"
        logger.info(f"File '{file_name}' uploaded successfully to S3 as '{s3_url}'")
    except ClientError as e:
        logger.error(f"Failed to upload {file_name}: {e}")
    return s3_url

# Log to PostgreSQL
def log_to_postgres(file_name, s3_url):
    """Log file and S3 URL to PostgreSQL."""
    with SessionLocal() as session:
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        session.add(log)
        session.commit()
        logger.info(f"Logged {file_name} with URL {s3_url} in PostgreSQL")

# Add documents to Neo4j
def add_graph_to_neo4j(graph_docs):
    with driver.session() as session:
        for doc in graph_docs:
            for node in doc.nodes:
                session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})
            for edge in doc.relationships:
                session.run(
                    "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                    "MERGE (source)-[:RELATED_TO {type: $type}]->(target)",
                    {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                )
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
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.emissions_tracker = EmissionsTracker(
            project_name="document_processing",
            log_level='warning'
        )
        self.tracker_active = False  # Nouveau drapeau pour l'état du tracker

    async def process_document(self, doc_converter: DocumentConverter, doc_path: Path) -> Optional[ConversionResult]:
        try:
            # Vérification de l'état du tracker et démarrage si nécessaire
            if not self.tracker_active:
                self.emissions_tracker.start()
                self.tracker_active = True

            start_time = time.time()

            with mlflow.start_run(nested=True) as run:
                mlflow.set_tag("device", str(self.device_manager.device))
                result = list(doc_converter.convert_all([doc_path], raises_on_error=False))[0]

                process_time = time.time() - start_time
                emissions = self.emissions_tracker.stop()
                self.tracker_active = False  # Remettre le drapeau à False
                CARBON_EMISSIONS.set(emissions if emissions else 0)

                # Log des métriques
                mlflow.log_metric("processing_time", process_time)
                mlflow.log_metric("carbon_emissions", emissions if emissions else 0)
                if result.status == ConversionStatus.SUCCESS:
                    mlflow.log_metric("success", 1)
                else:
                    mlflow.log_metric("failure", 1)

                self.device_manager.log_device_stats()

                logger.info(f"Document {doc_path.name} traité en {process_time:.2f}s")
                logger.info(f"Émissions CO2 estimées: {emissions:.4f}g" if emissions else "Émissions non mesurées")

                return result

        except Exception as e:
            logger.exception(f"Erreur lors du traitement de {doc_path}: {str(e)}")
            self.tracker_active = False  # Assurez-vous de remettre le drapeau à False en cas d'erreur
            return None

# Configuration FastAPI
app = FastAPI(title="Document and Graph-Based Retrieval API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tables_dir = OUTPUT_DIR / "tables"
figures_dir = OUTPUT_DIR / "figures"
tables_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Initialisation des gestionnaires
device_manager = DeviceManager()
model_manager = ModelManager(device_manager)

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"


# Custom docling options and class for processing documents
class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False

def create_document_converter(use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool) -> DocumentConverter:
    pipeline_options = CustomPdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = export_tables
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = export_figures
    if enrich_figures:
        pipeline_options.do_picture_classifier = enrich_figures

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.HTML, InputFormat.IMAGE],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)}
    )

# API Endpoints for document processing, indexing, and retrieval
@app.post("/process_document/")
async def process_document(files: List[UploadFile], use_ocr: bool = False, export_figures: bool = True, export_tables: bool = True, enrich_figures: bool = False):
    REQUEST_COUNT.inc()
    start_time = time.time()
    doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    results = []
    for file in files:
        temp_file = output_dir / file.filename
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        result = await model_manager.process_document(doc_converter, temp_file)
        if result:
            results.append(result)
            # Upload to S3 and log to PostgreSQL
            s3_url = upload_to_s3(temp_file, bucket_name)
            log_to_postgres(file.filename, s3_url)
    
    processing_time = time.time() - start_time
    PROCESS_TIME.observe(processing_time)
    return {"message": "Documents processed successfully", "time_taken": processing_time}

# Index documents into Neo4j
@app.post("/index_documents/")
def index_documents(folder_path: str):
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        add_graph_to_neo4j(graph_docs)
    return {"message": "Documents indexed successfully in Neo4j"}

# Search documents
# Graph data route: Fetch graph data from Neo4j
@app.get("/graph_data/")
def get_graph_data():
    logger.info("Fetching graph data from Neo4j.")
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
                RETURN e1.name AS source, e1.type AS source_type, e2.name AS target, e2.type AS target_type, r.type AS relationship
                """
            )
            graph_data = [
                {
                    "source": record["source"],
                    "source_type": record["source_type"],
                    "target": record["target"],
                    "target_type": record["target_type"],
                    "relationship": record["relationship"]
                }
                for record in result
            ]
        logger.info(f"Graph data retrieved: {graph_data}")
        return {"graph_data": graph_data}
    except Exception as e:
        logger.error(f"Error while fetching graph data from Neo4j: {str(e)}")
        return {"error": "An error occurred while fetching graph data."}

# List entities route: List all entities in Neo4j
@app.get("/list_entities/")
def list_entities():
    logger.info("Listing all entities in Neo4j.")
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.type AS type
                LIMIT 100
                """
            )
            entities = [{"name": record["name"], "type": record["type"]} for record in result]
        logger.info(f"Found entities: {entities}")
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Error while listing entities from Neo4j: {str(e)}")
        return {"error": "An error occurred while listing the entities."}

# Query Neo4j route: Query Neo4j for entities
@app.post("/query/")
def query_neo4j(query: str):
    logger.info(f"Querying Neo4j for: {query}")
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $query
                RETURN e.name AS name, e.type AS type
                """,
                {"query": query}
            )
            entities = [{"name": record["name"], "type": record["type"]} for record in result]
        logger.info(f"Query result: {entities}")
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Error while querying Neo4j: {str(e)}")
        return {"error": "An error occurred while querying the database."}

# Hybrid search route
@app.post("/search/")
def hybrid_search(query: str):
    index_name = "vector"
    keyword_index_name = "keyword"

    store = Neo4jVector.from_existing_index(
        ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name=index_name,
        keyword_index_name=keyword_index_name,
        search_type="hybrid",
    )

    retriever = store.as_retriever()
    results = retriever.invoke(query)

    return {"results": results}


prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions about IPM.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat endpoint using RAG with ChatOllama
@app.post("/chat/")
def chat(query: str):
    index_name = "vector"
    keyword_index_name = "keyword"

    store = Neo4jVector.from_existing_index(
        ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name=index_name,
        keyword_index_name=keyword_index_name,
        search_type="hybrid",
    )
    
    retriever = store.as_retriever()
    rag_chain = (
        {"context": retriever.invoke() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke({"question": query})
        return {"query": query, "response": response}
    except Exception as e:
        logger.error(f"Error during chat query: {str(e)}")
        return {"error": str(e)}

# Cypher QA route
@app.post("/cypher_query/")
def cypher_query(query: str):
    try:
        qa_chain = GraphCypherQAChain.from_llm(
            llm=llm, 
            graph=graph, 
            verbose=True,
            allow_dangerous_requests=True
        )
        
        result = qa_chain.invoke({"query": query})
        cypher_query = result.get('cypher_query', 'No Cypher query generated')
        final_response = result.get('result', 'No result generated')

        return {
            "query": query,
            "cypher_query": cypher_query,
            "response": final_response,
        }
    
    except Exception as e:
        logger.error(f"Error during Cypher QA: {str(e)}")
        return {"error": str(e)}

@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.json],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.json, ExportFormat.yaml, ExportFormat.md]
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR lors de la conversion."),
    export_figures: bool = Query(True, title="Exporter les figures", description="Activer ou désactiver l'exportation des figures"),
    export_tables: bool = Query(True, title="Exporter les tableaux", description="Activer ou désactiver l'exportation des tableaux"),
    enrich_figures: bool = Query(False, title="Enrichir les figures", description="Activer ou désactiver l'enrichissement des figures")
):
    REQUEST_COUNT.inc()
    logger.info(f"Réception de {len(files)} fichiers pour traitement")

    with mlflow.start_run(run_name="batch_processing"):
        start_time = time.time()
        input_file_paths = []
        doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)

        for file in files:
            if not file.filename.lower().endswith(('pdf', 'docx', 'pptx', 'html', 'jpg', 'jpeg', 'png')):
                raise HTTPException(
                    status_code=400,
                    detail="Format de fichier non supporté. Formats acceptés : PDF, DOCX, PPTX, HTML, JPG, JPEG, PNG"
                )

            temp_file = OUTPUT_DIR / file.filename
            async with aiofiles.open(temp_file, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            input_file_paths.append(temp_file)

        results = []
        for doc_path in input_file_paths:
            result = await model_manager.process_document(doc_converter, doc_path)
            if result:
                results.append(result)

        export_results = export_documents(results, OUTPUT_DIR, export_formats, export_figures, export_tables)

        total_time = time.time() - start_time
        PROCESS_TIME.observe(total_time)

        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("files_processed", len(results))

        return JSONResponse(content={
            "message": "Traitement terminé",
            "success_count": export_results[0],
            "partial_success_count": export_results[1],
            "failure_count": export_results[2],
            "processing_time": f"{total_time:.2f}s"
        })

@app.post("/upload_path/")
async def upload_path(
    file_path: str = Form(..., title="Chemin du dossier", description="Chemin vers le dossier contenant les fichiers"),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.json],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation"
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR pour la conversion."),
    export_figures: bool = Query(True, title="Exporter les figures", description="Activer ou désactiver l'exportation des figures"),
    export_tables: bool = Query(True, title="Exporter les tableaux", description="Activer ou désactiver l'exportation des tableaux"),
    enrich_figures: bool = Query(False, title="Enrichir les figures", description="Activer ou désactiver l'enrichissement des figures")
):
    REQUEST_COUNT.inc()
    logger.info(f"Traitement du dossier: {file_path}")
    input_dir_path = Path(file_path)

    if not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Le chemin spécifié n'est pas un répertoire valide.")

    # Extensions de fichiers supportées
    supported_extensions = ['.pdf', '.docx', '.pptx', '.html', '.jpg', '.jpeg', '.png']

    # Filtrer les fichiers avec les extensions supportées
    input_file_paths = [
        file for file in input_dir_path.glob('*')
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    if not input_file_paths:
        raise HTTPException(status_code=400, detail="Aucun fichier valide dans le répertoire.")

    with mlflow.start_run(run_name="directory_processing"):
        start_time = time.time()
        doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)

        results = []
        for doc_path in input_file_paths:
            result = await model_manager.process_document(doc_converter, doc_path)
            if result:
                results.append(result)

        export_results = export_documents(results, OUTPUT_DIR, export_formats, export_figures, export_tables)

        total_time = time.time() - start_time
        PROCESS_TIME.observe(total_time)

        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("files_processed", len(results))

        return JSONResponse(content={
            "success_count": export_results[0],
            "partial_success_count": export_results[1],
            "failure_count": export_results[2],
            "total_processed": len(results),
            "processing_time": f"{total_time:.2f}s"
        })

def export_documents(conv_results: List[ConversionResult], output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool) -> Tuple[int, int, int]:
    success_count = 0
    partial_success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1

            # Log additionnel pour vérifier le nombre d'images détectées
            num_figures = len([e for e in conv_res.document.iterate_items() if isinstance(e, PictureItem)])
            logger.info(f"Détection de {num_figures} figures dans le document {doc_filename}.")

            # Export des résultats dans les formats sélectionnés
        if ExportFormat.json in export_formats:
            json_path = output_dir / f"{doc_filename}.json"
            with json_path.open("w", encoding='utf-8') as json_file:
                json.dump(conv_res.document.export_to_dict(), json_file, indent=4)