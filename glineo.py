from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_postgres import PGVector
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from py2neo import Graph, NodeMatcher, Relationship
from fastapi import FastAPI, File, Request, Query, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pydantic import BaseModel
from typing import List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import aiofiles
import os
import re
import yaml
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()
app = FastAPI(title="Document Processing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.add("logs/conversion_{time}.log", rotation="1 day", retention="7 days", level="INFO")

# Metrics
Instrumentator().instrument(app).expose(app)
# Custom metrics
start_http_server(8002)

REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")

NEO4J_REQUEST_COUNT = Counter("neo4j_request_count", "Number of requests sent to Neo4j")
NEO4J_REQUEST_FAILURES = Counter("neo4j_request_failures", "Number of failed Neo4j requests")
NEO4J_REQUEST_LATENCY = Histogram("neo4j_request_latency_seconds", "Latency of Neo4j requests")

DOCUMENT_PROCESSING_SUCCESS = Counter("document_processing_success", "Number of successfully processed documents")
DOCUMENT_PROCESSING_FAILURES = Counter("document_processing_failures", "Number of failed document processing attempts")

# Neo4j setup
URI = "bolt://localhost:7687"
USER, PASSWORD = "neo4j", "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

@app.on_event("startup")
async def startup_event():
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "None"
logger.info(f"GPU détecté: {gpu_name}" if device.type == "cuda" else "Exécution sur CPU.")

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

class RetrievalQuery(BaseModel):
    query: str
    top_k: int = 5

def clean_text(text: str) -> str:
    """Clean up text to remove unwanted characters and normalize whitespace."""
    text = text.replace("\n", " ").strip()
    return re.sub(r'\s+', ' ', text)


def add_relationships(tx, relationships: List[Relationship]):
    """Add relationships to the Neo4j database."""
    for rel in relationships:
        try:
            if not rel.source or not rel.target or not rel.type:
                logger.warning(f"Skipping invalid relationship: {rel}")
                continue

            logger.info(f"Adding Relationship: {rel.type} ({rel.source.id} -> {rel.target.id})")
            tx.run(
                """
                MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
                MERGE (source)-[r:$type {properties: $properties}]->(target)
                ON CREATE SET r.created_at = timestamp()
                """,
                {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "type": rel.type,
                    "properties": rel.properties or {},
                },
            )
        except Exception as e:
            logger.error(f"Failed to add relationship: {rel}. Error: {e}")


def process_document(doc: Document):
    """Extraire et indexer un document dans Neo4j."""
    try:
        logger.info(f"Traitement du document : {doc.metadata.get('source', 'unknown')}")
        split_docs = text_splitter.split_documents([doc])
        logger.info(f"Document split into {len(split_docs)} chunks.")
        clean_docs = [Document(page_content=clean_text(chunk.page_content), metadata=chunk.metadata) for chunk in split_docs]
        graph_docs = graph_transformer.convert_to_graph_documents(clean_docs)
        doc_links = [gliner_extractor.extract_one(chunk) for chunk in clean_docs]

        with driver.session() as session:
            with session.begin_transaction() as tx:
                for graph_doc, links in zip(graph_docs, doc_links):
                    # Add nodes
                    if hasattr(graph_doc, "nodes") and graph_doc.nodes:
                        for node in graph_doc.nodes:
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
                            logger.info(f"Indexed Node: {node.id}, Type: {node.type}")

                    # Add relationships
                    if hasattr(graph_doc, "edges") and graph_doc.edges:
                        add_relationships(tx, graph_doc.edges)

                    # Add links
                    for link in links:
                        if not link.tag or not link.kind:
                            logger.warning(f"Skipping invalid link: {link}")
                            continue
                        logger.info(f"Adding Link: {link}")
                        tx.run(
                            """
                            MERGE (e:Entity {name: $name})
                            ON CREATE SET e.created_at = timestamp()
                            RETURN e
                            """,
                            {"name": link.tag},
                        )
    except Exception as e:
        logger.error(f"Error processing document: {doc.metadata.get('name', 'unknown')} - {e}")

def load_documents(file_path: Path, is_directory: bool = False) -> List[Document]:
    """Charger les documents depuis un fichier ou un dossier."""
    try:
        loader = PyPDFDirectoryLoader(str(file_path)) if is_directory else PyPDFLoader(str(file_path))
        return loader.load()
    except Exception as e:
        logger.error(f"Erreur de chargement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

async def handle_indexing(file: UploadFile = None, folder_path: str = None):
    """Centraliser la logique d'indexation."""
    try:
        documents = []
        if file:
            temp_file = OUTPUT_DIR / file.filename
            async with aiofiles.open(temp_file, "wb") as out_file:
                await out_file.write(await file.read())
            documents = load_documents(temp_file)
            temp_file.unlink()
        elif folder_path:
            folder = Path(folder_path)
            if not folder.is_dir():
                raise ValueError("Le chemin spécifié n'est pas un répertoire valide.")
            documents = load_documents(folder, is_directory=True)

        if not documents:
            raise ValueError("Aucun document valide trouvé.")

        logger.info(f"{len(documents)} documents chargés pour traitement.")
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_document, documents)

        return {"message": f"{len(documents)} documents indexés avec succès.", "gpu_used": gpu_name if device.type == "cuda" else "None"}
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")

# Routes 
@app.post("/index_documents/")
def index_documents(folder_path: str = "/home/pi/Documents/IF-SRV/4pdfs_subset/"):
    """
    Index documents from the given folder into Neo4j, extracting entities and relationships
    using GLiNER and GLiNERLinkExtractor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}. Execution will use the GPU.")
    else:
        logger.warning("No GPU detected. Execution will fall back to CPU.")

    logger.info(f"Indexing documents from folder: {folder_path}")
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    total_docs = len(documents)
    logger.info(f"Loaded {total_docs} documents for indexing.")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_document, documents)

    logger.info(f"Successfully indexed {total_docs} documents into Neo4j.")
    return {
        "message": f"{total_docs} documents indexed into Neo4j",
        "gpu_used": device.type == "cuda",
        "gpu_name": gpu_name if device.type == "cuda" else "None",
    }


@app.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    """Indexation d'un document unique."""
    return await handle_indexing(file=file)


@app.post("/index_document_path/")
async def index_document_path(folder_path: str):
    """Indexation de tous les documents d'un dossier."""
    return await handle_indexing(folder_path=folder_path)


@app.post("/verify_index/")
async def verify_index():
    """Vérification de l'état de la base de données Neo4j."""
    try:
        with driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN COUNT(n) AS count").single()["count"]
            total_relationships = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS count").single()["count"]
        return {"total_nodes": total_nodes, "total_relationships": total_relationships}
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'index: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la vérification de l'index.")

@app.post("/retrieve_documents/")
async def retrieve_documents(request: RetrievalQuery):
    # Perform the similarity search
    results = vectorstore.similarity_search(request.query, k=request.top_k)
    
    # Format and return results
    return {"results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]}

@app.post("/search/")
def hybrid_search(query: str):
    store = Neo4jVector.from_existing_index(
        embeddings=ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name="vector_index",
        keyword_index_name="keyword",
        search_type="hybrid",
    )
    retriever = store.as_retriever()
    results = retriever.invoke(query)
    return {"results": results}