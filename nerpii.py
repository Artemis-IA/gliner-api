import os
import time
import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import aiofiles
import boto3
import torch
import mlflow
import psutil
import GPUtil
from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter, Summary, Gauge, start_http_server
from neo4j import GraphDatabase
from sqlalchemy import create_engine, Column, String, Integer, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from transformers import AutoTokenizer
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from codecarbon import EmissionsTracker
from sqlalchemy.ext.declarative import declarative_base

# Database and MLflow setup
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)
mlflow.set_tracking_uri("http://localhost:5002")
start_http_server(8001)

# S3 Client setup
s3_client = boto3.client('s3', endpoint_url='http://127.0.1.1:9000',
                         aws_access_key_id='new_access_key',
                         aws_secret_access_key='new_secret_key')
bucket_name = 'pdfs'

# Neo4j setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Prometheus metrics setup
REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_LATENCY = Summary("request_latency_seconds", "Time spent processing request")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")

# Log structure
class ApiMlflowLog(Base):
    __tablename__ = 'api_mlflow_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    log_level = Column(String, nullable=False)
    service = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    message = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)

class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)
    mlflow_run_id = Column(String, ForeignKey("api_mlflow_logs.id"))
    mlflow_log = relationship("ApiMlflowLog", back_populates="document_logs")

ApiMlflowLog.document_logs = relationship("DocumentLog", back_populates="mlflow_log")

# PII Patterns
PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
}

# FastAPI app setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# GLiNER setup for PII
gliner_extractor = GLiNERLinkExtractor(labels=["PERSON", "EMAIL", "PHONE"], model="E3-JSI/gliner-multi-pii-domains-v1")

# Device Manager
class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def log_device_stats(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info()
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory.rss)
        logger.info(f"CPU Usage: {cpu_percent}% | Memory: {memory.rss / 1024 / 1024:.2f}MB")

# Upload and process document to S3
def upload_to_s3(file_name: str, bucket: str) -> str:
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
        return f"s3://{bucket}/{os.path.basename(file_name)}"
    except Exception as e:
        logger.error(f"Failed to upload {file_name}: {e}")
        return ""

def log_to_postgres(log_level: str, service: str, endpoint: str, message: str, parameters: Optional[Dict] = None, metrics: Optional[Dict] = None):
    with SessionLocal() as session:
        log_entry = ApiMlflowLog(log_level=log_level, service=service, endpoint=endpoint, message=message, parameters=parameters, metrics=metrics)
        session.add(log_entry)
        session.commit()

# Main API endpoints
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    REQUEST_COUNT.inc()
    device_manager = DeviceManager()
    start_time = time.time()
    doc_converter = DocumentConverter()
    
    results = []
    for file in files:
        temp_path = Path(file.filename)
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        result = process_document(doc_converter, temp_path)
        if result:
            s3_url = upload_to_s3(temp_path, bucket_name)
            log_to_postgres("INFO", "Document Processing API", "/upload", f"Uploaded and processed {file.filename}", {"s3_url": s3_url})

            # Save to Neo4j
            graph_transformer = GlinerGraphTransformer()
            graph_docs = graph_transformer.convert_to_graph_documents([result.document])
            add_graph_to_neo4j(graph_docs)
            results.append(result)
    
    mlflow.log_metric("processing_time", time.time() - start_time)
    return JSONResponse(content={"results": [str(r) for r in results]})

@app.post("/search/")
def hybrid_search(query: str):
    with mlflow.start_run(run_name="Search"):
        mlflow.log_param("query", query)
        store = Neo4jVector.from_existing_index(OllamaEmbeddings(), url=URI, username=USER, password=PASSWORD)
        retriever = store.as_retriever()
        results = retriever.invoke(query)
        mlflow.log_metric("result_count", len(results))
        log_to_postgres("INFO", "API", "/search", "Search completed", {"query": query}, {"result_count": len(results)})
    return {"results": results}

@app.post("/pii_remove/")
def pii_remove(folder_path: str = Form(...)):
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    processed_docs = []
    for doc in documents:
        redacted_content = redact_pii(doc.page_content)
        processed_docs.append({"original": doc.page_content, "redacted": redacted_content})
    return {"processed_documents": processed_docs}

def redact_pii(text: str) -> str:
    links = gliner_extractor.extract_one(text)
    for link in links:
        text = re.sub(r'\b' + re.escape(link.tag) + r'\b', "[REDACTED]", text)
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED {label}]", text)
    return text

# Neo4j insert
def add_graph_to_neo4j(graph_docs):
    with driver.session() as session:
        for doc in graph_docs:
            for node in doc.nodes:
                session.run("MERGE (e:Entity {name: $name, type: $type})", {"name": node.id, "type": node.type})
            for edge in doc.relationships:
                session.run("MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) MERGE (source)-[:RELATED_TO]->(target)",
                            {"source": edge.source.id, "target": edge.target.id})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
