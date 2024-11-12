from sqlalchemy import create_engine, Column, String, Integer, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from fastapi import FastAPI, Form, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from loguru import logger
from sqlalchemy import create_engine, Column, String, Integer, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3, os, re, mlflow, time, yaml
from prometheus_client import Counter, Summary

# Database setup
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Table for structured logging of API and MLflow logs
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

# Table for document logs
class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    file_name = Column(String, nullable=False)
    s3_url = Column(String, nullable=False)
    mlflow_run_id = Column(String, ForeignKey("api_mlflow_logs.id"))
    mlflow_log = relationship("ApiMlflowLog", back_populates="document_logs")

# Establish back relationship
ApiMlflowLog.document_logs = relationship("DocumentLog", back_populates="mlflow_log")

# Table for graph data logging in Neo4j
class Neo4jEntity(Base):
    __tablename__ = 'neo4j_entities'
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)

class Neo4jRelationship(Base):
    __tablename__ = 'neo4j_relationships'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey('neo4j_entities.id'), nullable=False)
    target_id = Column(Integer, ForeignKey('neo4j_entities.id'), nullable=False)
    relationship_type = Column(String, nullable=False)
    source = relationship("Neo4jEntity", foreign_keys=[source_id])
    target = relationship("Neo4jEntity", foreign_keys=[target_id])

# Table for PII logs
class ProcessedDocumentLog(Base):
    __tablename__ = 'processed_document_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    original_content = Column(String)
    anonymized_content = Column(String)
    doc_metadata = Column(JSON) 
# Ensure all tables are created in the database
Base.metadata.create_all(bind=engine)

# Prometheus and MLflow metrics setup
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# Initialize FastAPI
app = FastAPI(
    title="Document and Graph-Based Retrieval API",
    description="An API for indexing, querying, and analyzing documents with monitoring for model parameters and performance.",
    version="1.0.0"
)

# Neo4j and MLflow setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
mlflow.set_tracking_uri("http://localhost:5002")

# Initialize embeddings and model
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# S3 setup for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://127.0.1.1:9000',
    aws_access_key_id='new_access_key',
    aws_secret_access_key='new_secret_key'
)
bucket_name = 'pdfs'

# GLiNER configuration
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
gliner_extractor = GLiNERLinkExtractor(labels=config["labels"], model="E3-JSI/gliner-multi-pii-domains-v1")
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="knowledgator/gliner-multitask-large-v0.5",
    entity_confidence_threshold=0.1
)

def log_to_postgres(log_level, service, endpoint, message, parameters=None, metrics=None):
    with SessionLocal() as session:
        log_entry = ApiMlflowLog(
            log_level=log_level,
            service=service,
            endpoint=endpoint,
            message=message,
            parameters=parameters,
            metrics=metrics
        )
        session.add(log_entry)
        session.commit()

def upload_to_s3(file_name, bucket):
    try:
        s3_client.upload_file(file_name, bucket, os.path.basename(file_name))
        return f"s3://{bucket}/{os.path.basename(file_name)}"
    except Exception as e:
        logger.error(f"Failed to upload {file_name}: {e}")
        return None

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

# Endpoint to index documents
@app.post("/index/")
@REQUEST_LATENCY.time()
def index_pdfs(folder_path: Optional[str] = Form("/path/to/pdfs")):
    REQUEST_COUNT.inc()
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    start_time = time.time()
    with mlflow.start_run(run_name="Index PDFs"):
        for doc in documents:
            if not hasattr(doc, "page_content"):
                continue
            split_docs = text_splitter.split_documents([doc])
            graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
            add_graph_to_neo4j(graph_docs)

            file_name = doc.metadata.get("source", "unknown")
            s3_url = upload_to_s3(file_name, bucket_name)
            mlflow.log_param("file_name", file_name)
            mlflow.log_metric("doc_upload_success", int(s3_url is not None))
            
            if s3_url:
                log_to_postgres(
                    log_level="INFO",
                    service="API",
                    endpoint="/index",
                    message="File indexed and uploaded",
                    parameters={"file_name": file_name},
                    metrics={"upload_success": int(s3_url is not None)}
                )
        mlflow.log_metric("processing_time", time.time() - start_time)

    return {"message": "Documents indexed successfully"}

# Hybrid search endpoint
@app.post("/search/")
def hybrid_search(query: str):
    with mlflow.start_run(run_name="Search"):
        mlflow.log_param("query", query)
        
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
        
        mlflow.log_metric("result_count", len(results))
        log_to_postgres(
            log_level="INFO",
            service="API",
            endpoint="/search",
            message="Search completed",
            parameters={"query": query},
            metrics={"result_count": len(results)}
        )
    
    return {"results": results}
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat endpoint with MLflow tracking
@app.post("/chat/")
def chat(query: str):
    with mlflow.start_run(run_name="Chat"):
        mlflow.log_param("query", query)

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
        rag_chain = (
            {"context": retriever.invoke() | format_docs, "question": RunnablePassthrough()}
            | PromptTemplate(template="Your question: {question}")
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke({"question": query})
        
        mlflow.log_metric("response_length", len(response))
        log_to_postgres(
            log_level="INFO",
            service="API",
            endpoint="/chat",
            message="Chat response generated",
            parameters={"query": query},
            metrics={"response_length": len(response)}
        )
    
    return {"query": query, "response": response}

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# PII redaction setup
PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
}

def redact_pii(text: str) -> str:
    links = gliner_extractor.extract_one(text)
    for link in links:
        label = link.kind.split(':')[-1].upper()
        text = re.sub(r'\b' + re.escape(link.tag) + r'\b', f"[REDACTED {label}]", text)

    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED {label}]", text)

    return text

class ProcessedDocument(BaseModel):
    content: str
    anonymized_content: str
    metadata: Dict[str, Any]

@app.post("/pii_remove/")
def pii_remove(folder_path: str = Form(...)):
    REQUEST_COUNT.inc()
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    processed_docs = []

    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        for split_doc in split_docs:
            redacted_content = redact_pii(split_doc.page_content).strip()
            processed_docs.append(ProcessedDocument(
                content=split_doc.page_content,
                anonymized_content=redacted_content,
                metadata=split_doc.metadata
            ))
    
    return {"processed_documents": processed_docs}
