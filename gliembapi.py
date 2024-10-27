from fastapi import FastAPI, Form, Response
from typing import Optional
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from loguru import logger
import yaml
import os
from langsmith import Client as LangSmith
from prometheus_client import Counter, Summary

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# LangSmith client for logging chain runs
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
langsmith_client = LangSmith(api_key=LANGCHAIN_API_KEY)

app = FastAPI()

# Setup Neo4j connection
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)

# Initialize embeddings using Ollama for embeddings
ollama_emb = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2")

# Load configuration for GLiNER entity extraction
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

# Function to add graph nodes and edges to Neo4j
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

# Helper function for LangSmith logging
# Ajoutez une variable de configuration pour activer/désactiver LangSmith
ENABLE_LANGSMITH_LOGGING = "false" # os.getenv("ENABLE_LANGSMITH_LOGGING", "false").lower() == "true"

# Modifiez la fonction log_chain_run pour vérifier cette variable
def log_chain_run(chain_name: str, input_data: str, output_data: str, metadata: dict):
    if not ENABLE_LANGSMITH_LOGGING:
        logger.info(f"LangSmith logging is disabled for {chain_name}.")
        return
    try:
        langsmith_client.create_run(
            name=chain_name,
            inputs={"query": input_data},
            run_type="chain",
            outputs={"response": output_data},
            metadata=metadata,
        )
        logger.info(f"Logged {chain_name} run to LangSmith.")
    except Exception as e:
        logger.error(f"Error logging to LangSmith: {e}")

# Fonction pour supprimer toutes les données du graphe dans Neo4j
def clear_neo4j_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Toutes les données ont été supprimées de Neo4j.")

# Mise à jour de la route d'indexation avec suppression des données
@app.post("/index/")
@REQUEST_LATENCY.time()
def index_pdfs(
    folder_path: Optional[str] = Form(None)
):
    REQUEST_COUNT.inc()
    logger.info("Démarrage de l'indexation des PDFs.")

    # Suppression des données précédentes
    clear_neo4j_database()

    documents = []

    # Chargement des documents
    if folder_path:
        if os.path.isdir(folder_path):
            loader = PyPDFDirectoryLoader(folder_path)
            try:
                documents = loader.load()
                logger.info(f"{len(documents)} documents chargés.")
                
                # Ajouter 'source' aux métadonnées si absent
                for doc in documents:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = "default_source"  # ou un identifiant unique
            except Exception as e:
                logger.error(f"Erreur lors du chargement des documents: {e}")
                return {"error": f"Error loading documents: {str(e)}"}
        else:
            logger.error("Chemin de dossier invalide.")
            return {"error": "Invalid folder path"}
    else:
        logger.error("Aucun chemin de dossier fourni.")
        return {"error": "You must provide a folder path."}
    
    # Découpage des documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"{len(split_documents)} documents après découpage.")

    # Conversion en format graphe
    try:
        graph_documents = graph_transformer.convert_to_graph_documents(split_documents)
        logger.info("Transformation des documents en format graphe terminée.")
    except Exception as e:
        logger.error(f"Erreur lors de la transformation en graphe: {e}")
        return {"error": f"Error transforming documents: {str(e)}"}

    # Ajout des données graphe dans Neo4j (entités et relations)
    try:
        add_graph_to_neo4j(graph_documents)
        logger.info("Ajout des données graphe dans Neo4j terminé.")
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout des données graphe à Neo4j: {e}")
        return {"error": f"Error adding graph data to Neo4j: {str(e)}"}

    # Indexation hybride avec Ollama Embeddings dans Neo4j
    index_name = "vector"
    keyword_index_name = "keyword"

    try:
        store = Neo4jVector.from_documents(
            split_documents,
            embedding=ollama_emb,
            url=URI,
            username=USER,
            password=PASSWORD,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type="hybrid"
        )
        logger.info("Création ou récupération des index de recherche hybride terminée.")
    except Exception as e:
        logger.error(f"Erreur lors de la création des index: {e}")
        return {"error": f"Error creating indexes: {str(e)}"}

    return {"message": "Documents indexed successfully with vector and fulltext indexes."}

# Graph data route: Fetch graph data from Neo4j
@app.get("/graph_data/")
@REQUEST_LATENCY.time()
async def get_graph_data():
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

# List entities route: List all entities in Neo4j
@app.get("/list_entities/")
@REQUEST_LATENCY.time()
def list_entities():
    REQUEST_COUNT.inc()
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
@REQUEST_LATENCY.time()
def query_neo4j(query: str):
    REQUEST_COUNT.inc()
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
@REQUEST_LATENCY.time()
def hybrid_search(query: str):
    REQUEST_COUNT.inc()
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

    # Log search to LangSmith
    log_chain_run("Hybrid Search", query, results, {"query": query})

    return {"results": results}



# Prompt template with context
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Store session data
session_context = {}

@app.post("/chat/")
async def chat(query: str, session_id: str):
    # Retrieve context for the session or start new
    previous_context = session_context.get(session_id, "")
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
    # Define RAG pipeline using prompt and ChatOllama model
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm  # ChatOllama model
        | StrOutputParser()
    )

    # Execute pipeline
    response = chain.invoke({"context": previous_context, "question": query})
    
    # Update session context with the new response
    session_context[session_id] = previous_context + " " + query + " " + response['answer']

    return {"query": query, "response": response}

@app.post("/cyper_chat/")
async def chat(query: str):
    # Configurer le modèle et le retriever pour la chaîne
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
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Instancier le modèle de ChatOllama et la chaîne de RAG avec sources
    llm = ChatOllama(model="llama3.2")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    # Effectuer la requête avec la chaîne et obtenir la réponse
    response = chain.invoke({"question": query})

    # Retourner la réponse dans le format souhaité
    return {"query": query, "response": response}
    
# Cypher QA route
# Helper function to create a cypher prompt template
cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate Cypher query: {question}. Use `labels()` for node types, avoid `type()`."
)

# Initialize the GraphCypherQAChain with the graph and schema
qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    verbose=True
)

@app.post("/cypher_query/")
async def cypher_query(query: str):
    try:
        # Use the QA chain to generate and execute a Cypher query asynchronously
        result = await qa_chain.ainvoke({"query": query})
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

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)