from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import os
from loguru import logger

app = FastAPI()

# Neo4j connection setup
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Initialize GLiNER for entity extraction
gliner_extractor = GLiNERLinkExtractor(
    labels=[
        "Board of Directors", 
        "Audit Committee", 
        "Chief Financial Officer (CFO)", 
        "Chief Sustainability Officer", 
        "Governance and Compliance Committee", 
        "ESG Funds", 
        "Risk Manager", 
        "Institutional Investors", 
        "Audit Firms", 
        "Financial Regulatory Authorities", 
        "Sustainability Committee", 
        "Accounting Standards Organizations", 
        "Impact Investment Funds", 
        "Activist Shareholders", 
        "Corporate Governance Consultants", 
        "ESG Certification Bodies", 
        "Investor Relations Managers", 
        "Consumer Advocacy Groups", 
        "ESG Data Providers", 
        "Financial Analysts"
    ],
    model="urchade/gliner_mediumv2.1"
)

# Function to add entities and relationships to Neo4j
def add_to_neo4j(doc_id, entities, relationships):
    with driver.session() as session:
        # Create entities (nodes)
        for entity in entities:
            logger.info(f"Adding entity to Neo4j: {entity}")
            session.run(
                "MERGE (e:Entity {name: $name, type: $type})",
                {"name": entity.tag if hasattr(entity, 'tag') else getattr(entity, 'label', 'unknown'), "type": getattr(entity, 'kind', 'unknown')}
            )
        # Create relationships (edges)
        for rel in relationships:
            logger.info(f"Adding relationship to Neo4j: {rel}")
            session.run(
                """
                MATCH (source:Entity {name: $source}), (target:Entity {name: $target})
                MERGE (source)-[:RELATED_TO {type: $type}]->(target)
                """,
                {"source": rel["source"], "target": rel["target"], "type": rel["type"]}
            )


@app.get("/list_entities/")
async def list_entities():
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


@app.post("/index/")
async def index_pdfs(
    folder_path: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    documents = []

    # Handle folder path for directory input
    if folder_path:
        if os.path.isdir(folder_path):
            logger.info(f"Loading documents from folder: {folder_path}")
            loader = PyPDFDirectoryLoader(folder_path)
            documents = loader.load()
        else:
            logger.error("Invalid folder path provided.")
            return {"error": "Invalid folder path"}

    # Handle file uploads for individual files
    elif files:
        for file in files:
            file_path = f"./uploaded_files/{file.filename}"
            logger.info(f"Saving uploaded file: {file.filename}")
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            loader = PyPDFLoader(file_path)
            documents += loader.load()
    
    else:
        logger.error("No folder path or files provided for indexing.")
        return {"error": "You must provide either a folder path or files."}
    
    # Proceed with document processing
    logger.info("Splitting documents into smaller chunks.")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)  # Reduce chunk size to avoid truncation
    split_documents = text_splitter.split_documents(documents)

    for doc in split_documents:
        # Truncate content only if it exceeds the length allowed by GLiNER
        if len(doc.page_content) > 384:
            truncated_content = doc.page_content[:384]
            logger.warning(f"Document chunk truncated to 384 characters. Original length: {len(doc.page_content)}")
        else:
            truncated_content = doc.page_content

        logger.info(f"Extracting entities from document chunk: {truncated_content[:50]}...")
        try:
            # Using extract_one instead of extract_many for better control over each chunk
            entities = gliner_extractor.extract_one(truncated_content)
            relationships = []  # You should define how to extract relationships here.
            logger.info(f"Adding extracted entities and relationships to Neo4j for document ID: {doc.metadata.get('source', 'unknown_source')}")
            add_to_neo4j(doc.metadata.get('source', 'unknown_source'), entities, relationships)
        except Exception as e:
            logger.error(f"Error during entity extraction or Neo4j insertion: {str(e)}")
    
    logger.info("Documents indexed successfully.")
    return {"message": "Documents indexed successfully."}


# Question Answering Setup (with LLaMA)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

@app.post("/query/")
async def query_neo4j(query: str):
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
