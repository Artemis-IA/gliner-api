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
from langchain.chains import GraphCypherQAChain
from langchain_core.documents import Document as LCDocument
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
import argparse
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

# Initialize FastAPI
app = FastAPI(title="Unified Document Processing API", version="1.0")

# Setup Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER, NEO4J_PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# Setup PostgreSQL connection
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    s3_url = Column(String)

Base.metadata.create_all(bind=engine)

# Configure S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://127.0.1.1:9000',
    aws_access_key_id='new_access_key',
    aws_secret_access_key='new_secret_key'
)
bucket_name = 'pdfs'

# Ensure bucket exists or create it if it doesn't
try:
    s3_client.head_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' already exists.")
except ClientError:
    s3_client.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

# Initialize embeddings and model using Ollama
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# Fonction de chargement de configuration avec des valeurs par défaut
def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Ajoute une valeur par défaut pour max_entity_pair_distance si elle est absente
    config_dict.setdefault('max_entity_pair_distance', 100)
    return argparse.Namespace(**config_dict)

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
    relationship_confidence_threshold=0.1
)

# Function to upload a file to S3
def upload_to_s3(file_path: str, bucket: str) -> str:
    try:
        s3_client.upload_file(file_path, bucket, os.path.basename(file_path))
        s3_url = f"s3://{bucket}/{os.path.basename(file_path)}"
        logger.info(f"File '{file_path}' uploaded successfully to S3 as '{s3_url}'")
        return s3_url
    except ClientError as e:
        logger.error(f"Failed to upload {file_path}: {e}")
        return ""

# Log to PostgreSQL
def log_to_postgres(file_name: str, s3_url: str):
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

# Unified document upload and processing using Docling Markdown conversion
@app.post("/upload_and_process/")
async def upload_and_process(files: List[UploadFile] = File(None), folder_path: str = Form(None)):
    output_dir = "./temp"
    os.makedirs(output_dir, exist_ok=True)

    if folder_path:
        # Process files from a given folder path
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Invalid folder path")

        for root, _, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                s3_url = upload_to_s3(file_path, bucket_name)
                if not s3_url:
                    raise HTTPException(status_code=500, detail=f"Failed to upload file {filename} to S3")
                log_to_postgres(filename, s3_url)

                # Load and process document for Neo4j using Docling
                doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
                conversion_result = list(doc_converter.convert_all([file_path], raises_on_error=False))[0]
                if conversion_result.status == ConversionStatus.SUCCESS:
                    doc = conversion_result.document
                    markdown_text = doc.export_to_markdown()
                    loader = LCDocument(page_content=markdown_text)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents([loader])
                    graph_docs = graph_transformer.convert_to_graph_documents([LCDocument(page_content=split.page_content) for split in splits])
                    add_graph_to_neo4j(graph_docs)
    elif files:
        # Process uploaded files
        for file in files:
            file_path = os.path.join(output_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            s3_url = upload_to_s3(file_path, bucket_name)
            if not s3_url:
                raise HTTPException(status_code=500, detail="Failed to upload file to S3")

            log_to_postgres(file.filename, s3_url)

            # Load and process document for Neo4j using Docling
            doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
            conversion_result = list(doc_converter.convert_all([file_path], raises_on_error=False))[0]
            if conversion_result.status == ConversionStatus.SUCCESS:
                doc = conversion_result.document
                markdown_text = doc.export_to_markdown()
                loader = LCDocument(page_content=markdown_text)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents([loader])
                graph_docs = graph_transformer.convert_to_graph_documents([LCDocument(page_content=split.page_content) for split in splits])
                add_graph_to_neo4j(graph_docs)
    else:
        raise HTTPException(status_code=400, detail="No files or folder path provided")

    return {"message": "Documents uploaded, processed, and indexed successfully"}

# Endpoint for hybrid search
@app.post("/search/")
def hybrid_search(query: str):
    store = Neo4jVector.from_existing_index(
        ollama_emb,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid",
    )
    retriever = store.as_retriever()
    results = retriever.invoke(query)
    return {"results": results}

# Endpoint for question answering
document_prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions. Use the provided document excerpts to answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

@app.post("/qa/")
def qa(query: str):
    store = Neo4jVector.from_existing_index(
        ollama_emb,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid"
    )
    retriever = store.as_retriever()
    results = retriever.invoke(query)
    # Convert results to a structured format
    context = "\n".join([result['text'] for result in results])
    prompt = document_prompt_template.format(question=query, context=context, max_tokens=100)
    # Use the model to answer the question
    response = llm(prompt)
    return {"question": query, "answer": response}
