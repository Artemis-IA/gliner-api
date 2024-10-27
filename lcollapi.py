from fastapi import FastAPI, Form, HTTPException, Query, Response
from typing import Optional, List, Sequence
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated, TypedDict
from operator import add

from loguru import logger
import yaml
import os
from langsmith import Client as LangSmith
from prometheus_client import Counter, Summary

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# LangSmith client for logging chain runs
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
langsmith_client = LangSmith(api_key=LANGCHAIN_API_KEY)

app = FastAPI(
    title="Document and Graph-Based Retrieval API",
    description="An API for indexing, querying, and analyzing documents using embeddings and graph-based retrieval.",
    version="1.0.0"
)

# Setup Neo4j connection
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)

# Initialize embeddings and model using Ollama
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# GLiNER configuration
with open('gli_config.yml', 'r') as file:
    config = yaml.safe_load(file)

gliner_extractor = GLiNERLinkExtractor(
    labels=config["labels"],
    model="urchade/gliner_mediumv2.1"
)

graph_transformer = GlinerGraphTransformer(
    allowed_nodes=config["allowed_nodes"],
    allowed_relationships=config["allowed_relationships"],
    gliner_model="urchade/gliner_mediumv2.1",
    glirel_model="jackboyla/glirel_beta",
    entity_confidence_threshold=0.1,
    relationship_confidence_threshold=0.1,
)

# Add graph data to Neo4j
def add_graph_to_neo4j(graph_docs):
    with driver.session() as session:
        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                session.run(
                    "MERGE (e:Entity {name: $name, type: $type})",
                    {"name": node.id, "type": node.type}
                )
            if hasattr(graph_doc, 'relationships') and graph_doc.relationships:
                for edge in graph_doc.relationships:
                    session.run(
                        """
                        MATCH (source:Entity {name: $source}), (target:Entity {name: $target})
                        MERGE (source)-[:RELATED_TO {type: $type}]->(target)
                        """,
                        {"source": edge.source.id, "target": edge.target.id, "type": edge.type}
                    )

# Log chain runs to LangSmith
ENABLE_LANGSMITH_LOGGING = os.getenv("ENABLE_LANGSMITH_LOGGING", "false").lower() == "true"

def log_chain_run(chain_name: str, input_data: str, output_data: str, metadata: dict):
    if not ENABLE_LANGSMITH_LOGGING:
        return
    try:
        langsmith_client.create_run(
            name=chain_name,
            inputs={"query": input_data},
            run_type="chain",
            outputs={"response": output_data},
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Error logging to LangSmith: {e}")

# Clear Neo4j database
def clear_neo4j_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

@app.post("/index/", summary="Index documents", description="Index documents from a specified folder.")
@REQUEST_LATENCY.time()
def index_pdfs(folder_path: Optional[str] = Form(...)):
    """
    Index PDF documents located at the specified folder path.
    """
    REQUEST_COUNT.inc()
    clear_neo4j_database()

    documents = []
    if folder_path and os.path.isdir(folder_path):
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()
    else:
        raise HTTPException(status_code=400, detail="Invalid or missing folder path.")
    
    # Split and transform documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)
    graph_documents = graph_transformer.convert_to_graph_documents(split_documents)

    # Add data to Neo4j and index embeddings
    add_graph_to_neo4j(graph_documents)
    store = Neo4jVector.from_documents(
        split_documents,
        embedding=ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid"
    )
    return {"message": "Documents indexed successfully with vector and fulltext indexes."}

# Graph data route: Fetch graph data from Neo4j
@app.get("/graph_data/")
@REQUEST_LATENCY.time()
def get_graph_data():
    REQUEST_COUNT.inc()
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

@app.get("/list_entities/", summary="List entities", description="List all entities stored in Neo4j.")
@REQUEST_LATENCY.time()
def list_entities():
    """
    List all entities in Neo4j.
    """
    REQUEST_COUNT.inc()
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 100")
        entities = [{"name": record["name"], "type": record["type"]} for record in result]
    return {"entities": entities}

@app.post("/query/", summary="Query entities", description="Query entities from Neo4j based on a search string.")
@REQUEST_LATENCY.time()
def query_neo4j(query: str = Form(...)):
    """
    Query entities in Neo4j by name substring match.
    """
    REQUEST_COUNT.inc()
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.name AS name, e.type AS type",
            {"query": query}
        )
        entities = [{"name": record["name"], "type": record["type"]} for record in result]
    return {"entities": entities}

@app.post("/search/", summary="Hybrid search", description="Perform hybrid search using embeddings and keyword matching.")
@REQUEST_LATENCY.time()
def hybrid_search(query: str = Form(...)):
    """
    Perform a hybrid search in Neo4j using keyword and embedding matching.
    """
    REQUEST_COUNT.inc()
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
    return {"results": results}

# Define the State class for LangGraph
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# Define the call_model function
async def call_model(state: State):
    # Retrieve documents
    query = state["input"]
    retriever = Neo4jVector.from_existing_index(
        ollama_emb,
        url=URI,
        username=USER,
        password=PASSWORD,
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid"
    ).as_retriever(search_type="similarity", search_kwargs={"k": 4})

    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Prepare the prompt
    template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain_input = {
        "context": retrieved_text,
        "question": query
    }

    # Build the chain
    chain = prompt | llm

    # Get the response asynchronously
    ai_message = await chain.ainvoke(chain_input)
    
    # Extract the content from the AIMessage
    if isinstance(ai_message, AIMessage):
        response_text = ai_message.content
    else:
        response_text = str(ai_message)

    # Update the chat history
    new_chat_history = state.get("chat_history", []) + [
        HumanMessage(content=query),
        AIMessage(content=response_text)
    ]

    return {
        "chat_history": new_chat_history,
        "answer": response_text,
        "context": retrieved_text
    }

# Build the LangGraph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Compile the graph with a checkpointer
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

@app.post("/chat/", summary="Conversational chat", description="Chat with session-based memory.")
@REQUEST_LATENCY.time()
async def chat(query: str = Form(...), session_id: str = Form(...)):
    """
    Chat endpoint for conversations with session-based memory.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Invoke the graph asynchronously
    result = await app_graph.ainvoke({"input": query}, config=config)

    response = result["answer"]

    return {"query": query, "response": response}

@app.post("/cypher_query/", summary="Execute Cypher query", description="Run custom Cypher queries on Neo4j.")
@REQUEST_LATENCY.time()
async def cypher_query(query: str = Form(...)):
    """
    Execute a Cypher query in Neo4j.
    """
    with driver.session() as session:
        result = session.run(query)
        data = [record.data() for record in result]
    return {"query": query, "data": data}

@app.get("/metrics", summary="Metrics", description="Retrieve Prometheus metrics.")
async def metrics():
    """
    Get Prometheus metrics for API monitoring.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
