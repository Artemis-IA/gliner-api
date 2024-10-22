from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import os

app = FastAPI()

# Neo4j connection setup
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Initialize GLiNER for entity extraction
gliner_extractor = GLiNERLinkExtractor(
    labels = [
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
            session.run(
                "MERGE (e:Entity {name: $name, type: $type})",
                {"name": entity.tag, "type": entity.kind}
            )
        # Create relationships (edges)
        for rel in relationships:
            session.run(
                """
                MATCH (source:Entity {name: $source}), (target:Entity {name: $target})
                MERGE (source)-[:RELATED_TO {type: $type}]->(target)
                """,
                {"source": rel["source"], "target": rel["target"], "type": rel["type"]}
            )

@app.post("/index/")
async def index_pdfs(
    folder_path: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    documents = []

    # Handle folder path for directory input
    if folder_path:
        if os.path.isdir(folder_path):
            loader = PyPDFDirectoryLoader(folder_path)
            documents = loader.load()
        else:
            return {"error": "Invalid folder path"}

    # Handle file uploads for individual files
    elif files:
        for file in files:
            file_path = f"./uploaded_files/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            loader = PyPDFLoader(file_path)
            documents += loader.load()
    
    else:
        return {"error": "You must provide either a folder path or files."}
    
    # Proceed with document processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)

    for doc in split_documents:
        entities = gliner_extractor.extract_one(doc.page_content)
        relationships = []  # You should define how to extract relationships here.
        add_to_neo4j(doc.metadata.get('source', 'unknown_source'), entities, relationships)
    
    return {"message": "Documents indexed successfully."}

# Question Answering Setup (with LLaMA)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

@app.post("/query/")
async def query_neo4j(query: str):
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
    return {"entities": entities}
