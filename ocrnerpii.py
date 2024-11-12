from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor, LinkExtractorTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from loguru import logger
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3, os, re, yaml, cv2
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import numpy as np
from difflib import SequenceMatcher
from prometheus_client import Counter, Summary
from paddleocr import PaddleOCR
import textract

# Initialize FastAPI
app = FastAPI(
    title="Document and Graph-Based Retrieval API",
    description="An API for indexing, querying, and analyzing documents with enhanced OCR, PII masking, MLOps integrations, and graph-based entity extraction.",
    version="8.0.0"
)

# Prometheus metrics for tracking performance
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# Neo4j connection setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)

# Ollama embeddings and model setup
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# PostgreSQL connection setup for document logs
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class DocumentLog(Base):
    __tablename__ = 'document_logs'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    s3_url = Column(String)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# MinIO configuration for storing documents
s3_client = boto3.client(
    's3',
    endpoint_url='http://127.0.1.1:9000',
    aws_access_key_id='new_access_key',
    aws_secret_access_key='new_secret_key'
)
bucket_name = 'pdfs'
try:
    s3_client.head_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' already exists.")
except:
    s3_client.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

# GLiNER configuration for entity linking and PII extraction
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
with open('conf/pii_patterns.yml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)
    PII_PATTERNS = data['PII_PATTERNS']

gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="E3-JSI/gliner-multi-pii-domains-v1"
)
link_transformer = LinkExtractorTransformer([gliner_extractor])

graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="E3-JSI/gliner-multi-pii-domains-v1",
    glirel_model="jackboyla/glirel_beta",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="fr", use_gpu=True)

# Helper Functions

def clean_text(text):
    """Cleans and normalizes text for easier processing"""
    logger.debug("Cleaning text for normalization.")
    text = re.sub(r'(?<!\.\n)(?<!\!\n)(?<!\?\n)\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def mask_pii(text: str) -> str:
    """Masks all identified PII in the provided text"""
    logger.debug("Masking PII in the text.")
    # Mask PII with GLiNER
    links = gliner_extractor.extract_one(text)
    for link in links:
        label = link.kind.split(':')[-1].upper()
        pii_placeholder = f"[REDACTED {label}]"
        pattern = re.escape(link.tag)
        text = re.sub(r'\b' + pattern + r'\b', pii_placeholder, text)

    # Mask additional PII with regex patterns
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED {label.upper()}]", text)

    return text

def extract_text_with_paddle(image_path: str) -> str:
    """Extracts text from an image using PaddleOCR"""
    logger.info(f"Extracting text from image using PaddleOCR: {image_path}")
    result = ocr.ocr(image_path)
    return " ".join([line[1][0] for line in result])

def extract_text_with_tesseract(image_path: str) -> str:
    """Extracts text from an image using Tesseract OCR"""
    logger.info(f"Extracting text from image using Tesseract: {image_path}")
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF using pdfplumber and pdf2image"""
    logger.info(f"Extracting text from PDF using pdfplumber and pdf2image: {pdf_path}")
    texts = []
    
    # Use pdfplumber to extract simple text
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(clean_text(page_text))

    # Fallback to OCR with pdf2image if pdfplumber fails
    if not texts:
        logger.info("Falling back to OCR for extracting text from images")
        pages = convert_from_path(pdf_path, dpi=300)
        for page in pages:
            image_path = f"/tmp/page_{page.page_number}.jpg"
            page.save(image_path, 'JPEG')
            texts.append(extract_text_with_tesseract(image_path))

    return " ".join(texts)

def combine_ocr_results(paddle_text: str, tesseract_text: str) -> str:
    """Combines OCR results from PaddleOCR and Tesseract"""
    logger.debug("Combining OCR results from PaddleOCR and Tesseract.")
    if SequenceMatcher(None, paddle_text, tesseract_text).ratio() > 0.9:
        return paddle_text
    else:
        return paddle_text if len(paddle_text) > len(tesseract_text) else tesseract_text

def extract_tables_with_pdfplumber(pdf_path):
    """Extracts tables from a PDF using pdfplumber"""
    tables = []
    logger.info(f"Extracting tables from PDF using pdfplumber: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            tables.extend(page_tables)
    return tables

def process_document(file_path: str) -> Dict[str, Any]:
    """Processes a document for text extraction, PII masking, figure detection, and relationship extraction"""
    logger.info(f"Processing document: {file_path}")
    file_ext = file_path.split(".")[-1].lower()
    processed_data = {"original_text": "", "masked_text": "", "tables": [], "figures": []}

    if file_ext == "pdf":
        logger.debug(f"Processing PDF file: {file_path}")
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(raw_text)
        masked_text = mask_pii(cleaned_text)
        processed_data["original_text"] = cleaned_text
        processed_data["masked_text"] = masked_text

        # Extract structured tables
        processed_data["tables"] = extract_tables_with_pdfplumber(file_path)

    elif file_ext in ["jpg", "png"]:
        logger.debug(f"Processing image file: {file_path}")
        # Extract text using both PaddleOCR and Tesseract
        paddle_text = extract_text_with_paddle(file_path)
        tesseract_text = extract_text_with_tesseract(file_path)
        combined_text = combine_ocr_results(paddle_text, tesseract_text)
        
        # Clean and mask the text
        cleaned_text = clean_text(combined_text)
        masked_text = mask_pii(cleaned_text)
        
        processed_data["original_text"] = cleaned_text
        processed_data["masked_text"] = masked_text

    # Link extraction and transformation
    document_obj = Document(page_content=cleaned_text)
    processed_data["entities"] = extract_entities_and_links(document_obj)

    return processed_data

def extract_entities_and_links(document: Document) -> Dict[str, Any]:
    """Extract entities and relationships using GLiNER and LinkExtractorTransformer"""
    logger.debug("Extracting entities and links from document.")
    # Extract entities using GLiNERLinkExtractor
    links = gliner_extractor.extract_one(document)
    
    # Wrap the document with links added to metadata
    document.metadata['links'] = list(links)

    # Transform the documents using LinkExtractorTransformer
    transformed_documents = link_transformer.transform_documents([document])

    # Apply graph transformer to build graph-based relationships
    graph_data = graph_transformer.convert_to_graph_documents(transformed_documents)

    return graph_data

# API Endpoints

@app.post("/process_document/", summary="Process a document or directory for text extraction, PII masking, and entity analysis")
@REQUEST_LATENCY.time()
def process_document_endpoint(file_path: str = Form(...)):
    """Endpoint to handle document or directory processing requests"""
    REQUEST_COUNT.inc()
    logger.info(f"Received request to process: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Invalid path provided: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid path provided")

    # Check if it's a file or directory
    if os.path.isfile(file_path):
        # Process a single file
        result = process_document(file_path)
        logger.info(f"Successfully processed file: {file_path}")
        return {"processed_data": result}

    elif os.path.isdir(file_path):
        # Process all files in the directory
        all_results = {}
        for root, dirs, files in os.walk(file_path):
            for file in files:
                full_path = os.path.join(root, file)
                all_results[file] = process_document(full_path)

        logger.info(f"Successfully processed directory: {file_path}")
        return {"processed_data": all_results}
    
    else:
        logger.error("The specified path is neither a file nor a directory.")
        raise HTTPException(status_code=400, detail="The specified path is neither a file nor a directory.")
