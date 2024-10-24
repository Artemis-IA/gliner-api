from fastapi import FastAPI, Form
from typing import Optional
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from loguru import logger
import yaml
import os
import torch

app = FastAPI()

# Neo4j connection setup
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)


with open('gli_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Initialize GLiNER for entity extraction (for the chat route)
gliner_extractor = GLiNERLinkExtractor(
    labels=config["gliner"]["labels"],
    model="urchade/gliner_mediumv2.1"
)

# GlinerGraphTransformer setup
graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["gliner"]["allowed_nodes"],
    allowed_relationships=config["gliner"]["allowed_relationships"],
    gliner_model="urchade/gliner_mediumv2.1",
    glirel_model="jackboyla/glirel_beta",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Function to add graph nodes and edges to Neo4j
def add_graph_to_neo4j(graph_docs):
    with driver.session() as session:
        for graph_doc in graph_docs:
            # Add nodes to Neo4j
            for node in graph_doc.nodes:
                logger.info(f"Adding node to Neo4j: {node}")
                session.run(
                    "MERGE (e:Entity {name: $name, type: $type})",
                    {"name": node.id, "type": node.type}
                )
            # Add edges to Neo4j if they exist
            if hasattr(graph_doc, 'relationships') and graph_doc.relationships:
                for edge in graph_doc.relationships:
                    logger.info(f"Adding edge to Neo4j: {edge}")
                    session.run(
                        """
                        MATCH (source:Entity {name: $source}), (target:Entity {name: $target})
                        MERGE (source)-[:RELATED_TO {type: $type}]->(target)
                        """,
                        {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                    )



@app.post("/index/")
async def index_pdfs(
    folder_path: Optional[str] = Form(None)
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
    else:
        logger.error("No folder path provided for indexing.")
        return {"error": "You must provide a folder path."}
    
    # Proceed with document processing
    logger.info("Splitting documents into smaller chunks.")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)

    # Transform the split documents into graph format
    logger.info("Transforming documents to graph format using GlinerGraphTransformer.")
    graph_documents = graph_transformer.convert_to_graph_documents(split_documents)

    # Add to Neo4j
    logger.info("Adding graph entities and relationships to Neo4j.")
    add_graph_to_neo4j(graph_documents)

    logger.info("Documents indexed successfully.")
    return {"message": "Documents indexed successfully."}

@app.get("/graph_data/")
async def get_graph_data():
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


# LLaMA 3.2 Setup for Chat and Q&A
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create HuggingFace pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True, max_new_tokens=100, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Fix the prompt template for Cypher query generation
cypher_generation_template = """
Generate a Cypher query to retrieve relevant information from the graph database.
Schema: {schema}
Question: {question}
Cypher query:
"""
cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_generation_template)

# GraphCypherQAChain for Neo4j Queries, with allow_dangerous_requests=True
qa_chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True, 
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True  # Important to allow Cypher queries with potentially high risks
)

@app.post("/chat/")
async def chat(query: str):
    """
    Endpoint to handle chat queries using natural language to query Neo4j.
    """
    try:
        # Log the incoming query
        logger.info(f"Processing query using Neo4j and LLaMA: {query}")
        
        # Step 1: Extract the context and response using Neo4j
        result = qa_chain.invoke({"query": query})
        
        # Log the full result for debugging purposes
        logger.info(f"Full result: {result}")

        # Check if 'intermediate_steps' is available before accessing it
        if 'intermediate_steps' in result:
            cypher_query = result['intermediate_steps'][0]['query']
            graph_result = result['intermediate_steps'][1].get('context', 'No context available')
        else:
            cypher_query = "No Cypher query generated"
            graph_result = "No graph result available"
        
        # Step 2: Generate the final response using LLaMA
        final_response = result.get('result', 'No result generated')

        return {
            "query": query,
            "response": final_response,
            "cypher_query": cypher_query,  # The Cypher query generated
            "graph_result": graph_result,  # Data retrieved from the graph
        }
    
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        return {"error": str(e)}