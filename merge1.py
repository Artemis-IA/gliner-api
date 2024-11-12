import os
import re
import json
import random
import shutil
import zipfile
import threading
import yaml
import cv2
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import numpy as np
from difflib import SequenceMatcher
import torch
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Union, Optional, Any, Sequence
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response
from pydantic import BaseModel, Field
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.data_processing import GLiNERDataset
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
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from loguru import logger
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Summary
from paddleocr import PaddleOCR
import textract

# Initialize FastAPI
app = FastAPI(
    title="Unified API",
    description="An API that combines GLiNER functionalities, document indexing, graph-based retrieval, and document processing with OCR and PII masking.",
    version="1.0.0"
)

# Ensure directories exist
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("data"):
    os.makedirs("data")

# Device configuration
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Global variables
annotator = None
model_generator = None

# Available models
AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

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

# S3 client configuration for MinIO
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
except ClientError:
    s3_client.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

# GLiNER configuration
with open('conf/gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)
with open('conf/pii_patterns.yml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)
    PII_PATTERNS = data['PII_PATTERNS']

gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="E3-JSI/gliner-multi-pii-domains-v1"
)

graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="E3-JSI/gliner-multi-pii-domains-v1",
    glirel_model="jackboyla/glirel_beta",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)

link_transformer = LinkExtractorTransformer([gliner_extractor])

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="fr", use_gpu=True)

# Prometheus metrics for tracking performance
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., example="IBM Watson defeated human champions in the game of Jeopardy!")

class NERInput(BaseModel):
    model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    text: str = Field(..., example="IBM Watson defeated human champions in the game of Jeopardy!")
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
    model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    custom_model_name: str = Field(..., example="my-custom-model")
    train_data: str  # Path to the training data
    split_ratio: float = Field(0.9, example=0.9)
    learning_rate: float = Field(5e-6, example=5e-6)
    weight_decay: float = Field(0.01, example=0.01)
    batch_size: int = Field(8, example=8)
    epochs: int = Field(1, example=1)
    compile_model: bool = Field(False, example=False)

class EvaluateInput(BaseModel):
    model_name: str = Field(..., example="my-custom-model")

class EvaluateOutput(BaseModel):
    f1_score: float = Field(..., example=0.85)
    results: str = Field(..., example="Entity-wise F1 score: ...")

class ProcessedDocument(BaseModel):
    content: str
    anonymized_content: str
    metadata: Dict[str, Any]

class PiiRemoveResponse(BaseModel):
    processed_documents: List[ProcessedDocument]

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
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }
    r["entities"] = merge_entities(r["entities"])
    return r

class AutoAnnotator:
    def __init__(
        self, model_name: str = "knowledgator/gliner-multitask-large-v0.5",
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ) -> None:
        self.model = GLiNER.from_pretrained(model_name).to(device)
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
    links = gliner_extractor.extract_one(document.page_content)
    
    # Wrap the document with links added to metadata
    document.metadata['links'] = list(links)

    # Transform the documents using LinkExtractorTransformer
    transformed_documents = link_transformer.transform_documents([document])

    # Apply graph transformer to build graph-based relationships
    graph_data = graph_transformer.convert_to_graph_documents(transformed_documents)

    return graph_data

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

def log_to_postgres(file_name, s3_url):
    """Log file and S3 URL to PostgreSQL."""
    with SessionLocal() as session:
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        session.add(log)
        session.commit()
        logger.info(f"Logged {file_name} with URL {s3_url} in PostgreSQL")

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

# API Endpoints

# Endpoint from the first API
@app.post("/ner/", response_model=NEROutput)
def ner_endpoint(input_data: NERInput):
    model_path = f"models/{input_data.model_name}"
    if not os.path.exists(model_path):
        if input_data.model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(input_data.model_name).to(device)
        else:
            raise HTTPException(status_code=404, detail="Model not found.")
    else:
        model = GLiNER.from_pretrained(model_path).to(device)

    labels = [label.strip() for label in input_data.labels.split(",")] if input_data.labels else None
    result = annotate_text(
        model, input_data.text, labels, input_data.threshold, input_data.nested_ner
    )
    return result

@app.post("/annotate/")
def annotate_endpoint(input_data: AnnotateInput):
    try:
        labels = [label.strip() for label in input_data.labels.split(",")]
        annotator = AutoAnnotator(input_data.model)
        annotated_data = annotator.auto_annotate(
            input_data.sentences, labels, input_data.prompt, input_data.threshold
        )
        with open("data/annotated_data.json", "wt") as file:
            json.dump(annotated_data, file)
        return {"message": "Successfully annotated and saved as data/annotated_data.json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_dataset/")
def upload_dataset(file: UploadFile = File(...)):
    save_path = os.path.join("data", file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"File saved to {save_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
def train_endpoint(train_input: TrainInput):
    def load_and_prepare_data(train_path, split_ratio):
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"The file {train_path} does not exist.")

        with open(train_path, "r") as f:
            data = json.load(f)
        random.seed(42)
        random.shuffle(data)
        train_data = data[:int(len(data) * split_ratio)]
        test_data = data[int(len(data) * split_ratio):]
        return train_data, test_data

    def create_models_directory():
        if not os.path.exists("models"):
            os.makedirs("models")

    try:
        create_models_directory()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if train_input.model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(train_input.model_name)
        else:
            model_path = f"models/{train_input.model_name}"
            if os.path.exists(model_path):
                model = GLiNER.from_pretrained(model_path)
            else:
                raise HTTPException(status_code=404, detail="Model not found.")

        train_data, test_data = load_and_prepare_data(train_input.train_data, train_input.split_ratio)

        with open("data/test.json", "wt") as file:
            json.dump(test_data, file)

        train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
        test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
        data_collator = DataCollatorWithPadding(model.config)

        if train_input.compile_model:
            torch.set_float32_matmul_precision('high')
            model.to(device)
            model.compile_for_training()
        else:
            model.to(device)

        training_args = TrainingArguments(
            output_dir="models",
            learning_rate=train_input.learning_rate,
            weight_decay=train_input.weight_decay,
            others_lr=train_input.learning_rate,
            others_weight_decay=train_input.weight_decay,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            per_device_train_batch_size=train_input.batch_size,
            per_device_eval_batch_size=train_input.batch_size,
            num_train_epochs=train_input.epochs,
            evaluation_strategy="epoch",
            save_steps=1000,
            save_total_limit=10,
            dataloader_num_workers=8,
            use_cpu=(device == torch.device('cpu')),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        model.save_pretrained(f"models/{train_input.custom_model_name}")

        return {"message": "Training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/", response_model=EvaluateOutput)
def evaluate_endpoint(evaluate_input: EvaluateInput):
        try:
            model_path = f"models/{evaluate_input.model_name}"
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model not found.")

            model = GLiNER.from_pretrained(model_path).to(device)
            test_data_path = "data/test.json"
            if not os.path.exists(test_data_path):
                raise HTTPException(status_code=404, detail="Test data not found.")

            with open(test_data_path, "r") as file:
                    test_data = json.load(file)

            test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
            data_collator = DataCollatorWithPadding(model.config)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir="models",
                    per_device_eval_batch_size=8,
                    dataloader_num_workers=8,
                    use_cpu=(device == torch.device('cpu')),
                    report_to="none",
                ),
                eval_dataset=test_dataset,
                tokenizer=model.data_processor.transformer_tokenizer,
                data_collator=data_collator,
            )

            eval_results = trainer.evaluate()
            f1_score = eval_results.get("eval_f1", 0.0)
            results_str = json.dumps(eval_results, indent=2)

            return EvaluateOutput(f1_score=f1_score, results=results_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_document/", response_model=PiiRemoveResponse)
def process_document_endpoint(file: UploadFile = File(...)):
    try:
        save_path = os.path.join("data", file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed_data = process_document(save_path)
        s3_url = upload_to_s3(save_path, bucket_name)
        log_to_postgres(file.filename, s3_url)
        add_graph_to_neo4j(processed_data["entities"])

        processed_documents = [
            ProcessedDocument(
                content=processed_data["original_text"],
                anonymized_content=processed_data["masked_text"],
                metadata={"tables": processed_data["tables"], "figures": processed_data["figures"]}
            )
        ]

        return PiiRemoveResponse(processed_documents=processed_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
