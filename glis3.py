from fastapi import FastAPI, Form, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List, Sequence, Dict, Any
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
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3, os, threading, yaml, mlflow
from botocore.exceptions import ClientError
from prometheus_client import Counter, Summary
import re

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# Initialize FastAPI
app = FastAPI(
    title="Document and Graph-Based Retrieval API",
    description="An API for indexing, querying, and analyzing documents using embeddings and graph-based retrieval.",
    version="1.0.0"
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
    endpoint_url='http://127.0.1.1:9000',
    aws_access_key_id = 'new_access_key',
    aws_secret_access_key = 'new_secret_key'
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
    glirel_model="jackboyla/glirel_beta",
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

# Endpoint to index documents
@app.post("/index/", summary="Index documents", description="Index documents from a specified folder.")
@REQUEST_LATENCY.time()
def index_pdfs(folder_path: Optional[str] = Form("/home/pi/Documents/IF-SRV/4pdfs_subset/")):
    REQUEST_COUNT.inc()
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    for doc in documents:
        if not hasattr(doc, "page_content"):
            logger.error(f"Invalid document format: {doc}")
            continue
        split_docs = text_splitter.split_documents([doc])

        # Graph and S3 process
        graph_docs = graph_transformer.convert_to_graph_documents(split_docs)
        add_graph_to_neo4j(graph_docs)

        # S3 upload and logging
        s3_url = upload_to_s3(doc.metadata.get('source'), bucket_name)
        log_to_postgres(doc.metadata.get('source'), s3_url)

    return {"message": "Documents indexed and logged successfully"}


@app.get("/metrics")
async def metrics():
    # Your existing metrics logic here
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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

# Setup for LLM using Ollama chat model for RAG
llm = ChatOllama(model="llama3.2")

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
    
    
# Enhanced PII patterns for regex-based redaction
PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    "SIRET": r"\b\d{14}\b",
    "ADDRESS": r"\d{1,5}\s\w+\s\w+",
    "COMPANY": r"\b[A-Z][A-Z0-9&\.\- ]{2,}\b"
}

def redact_pii(text: str) -> str:
    """
    Redacts PII in the text using GLiNER and regex patterns.
    Replaces PII with placeholders.
    """
    # Use GLiNER to detect PII
    links = gliner_extractor.extract_one(text)
    logger.debug(f"PII détectés pour redaction: {[link.tag for link in links]}")

    for link in links:
        label = link.kind.split(':')[-1].upper()  # Extract label without prefix
        pii_placeholder = f"[REDACTED {label}]"
        # Use regex with word boundaries to replace exact matches
        pattern = re.escape(link.tag)
        text = re.sub(r'\b' + pattern + r'\b', pii_placeholder, text)

    # Redact remaining PII with regex
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED {label.upper()}]", text)

    return text

def remove_pii(text: str) -> str:
    """
    Removes PII from the text using GLiNER and regex patterns.
    Replaces PII with a space to preserve word separation.
    """
    # Use GLiNER to detect PII
    links = gliner_extractor.extract_one(text)
    logger.debug(f"PII détectés pour suppression: {[link.tag for link in links]}")

    for link in links:
        # Replace PII with a space
        pattern = re.escape(link.tag)
        text = re.sub(r'\b' + pattern + r'\b', ' ', text)

    # Remove remaining PII with regex
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, ' ', text)

    # Normalize whitespace to prevent word merging
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class ProcessedDocument(BaseModel):
    content: str
    anonymized_content: str
    metadata: Dict[str, Any]

class PiiRemoveResponse(BaseModel):
    processed_documents: List[ProcessedDocument]

@app.post("/pii_remove/", response_model=PiiRemoveResponse)
@REQUEST_LATENCY.time()
def pii_remove(folder_path: str = Form(...)):
    """
    Endpoint to remove PII from PDF documents in a specified folder.
    Returns both the cleaned content and the anonymized content.
    """
    REQUEST_COUNT.inc()

    if not os.path.isdir(folder_path):
        logger.error(f"Chemin de dossier invalide: {folder_path}")
        raise HTTPException(status_code=400, detail="Chemin de dossier invalide.")

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    redacted_docs = []

    for doc in documents:
        if not hasattr(doc, "page_content"):
            logger.warning(f"Document sans 'page_content': {doc}")
            continue
        split_docs = text_splitter.split_documents([doc])
        for split_doc in split_docs:
            original_text = split_doc.page_content

            # Normalize whitespace before processing
            original_text = re.sub(r'\s+', ' ', original_text)

            # Generate redacted and clean versions
            redacted_content = redact_pii(original_text).strip()
            clean_content = remove_pii(original_text).strip()

            redacted_docs.append(ProcessedDocument(
                content=clean_content,
                anonymized_content=redacted_content,
                metadata=split_doc.metadata
            ))

    logger.info(f"Traitement terminé: {len(redacted_docs)} documents")
    return PiiRemoveResponse(processed_documents=redacted_docs)
