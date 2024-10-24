from fastapi import FastAPI, Form, Response, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.graph_transformers.gliner import GlinerGraphTransformer
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from neo4j import GraphDatabase, Transaction
from langchain_core.prompts import PromptTemplate
from loguru import logger
import yaml
import os
from datetime import datetime
from langsmith import Client as LangSmith
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

# Models for request/response
class DocumentBase(BaseModel):
    title: str
    content: str
    metadata: Dict[str, Any] = {}

class DocumentCreate(DocumentBase):
    folder_path: str

class Document(DocumentBase):
    id: str
    created_at: datetime
    updated_at: datetime

class EntityBase(BaseModel):
    name: str
    type: str
    properties: Dict[str, Any] = {}

class EntityCreate(EntityBase):
    pass

class Entity(EntityBase):
    id: str

class RelationshipCreate(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = {}

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    include_metadata: bool = True

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent processing request')

# Initialize FastAPI app
app = FastAPI(title="Document Graph API", 
             description="API for document processing with Neo4j, LangChain, and Ollama",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "your_langsmith_key")

# Initialize clients
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
langsmith_client = LangSmith(api_key=LANGSMITH_API_KEY)
ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2")

# Load GLiNER configuration
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

# Neo4j Helper Functions
class Neo4jCRUD:
    @staticmethod
    async def create_document(tx: Transaction, doc: DocumentCreate) -> str:
        result = tx.run("""
            CREATE (d:Document {
                id: randomUUID(),
                title: $title,
                content: $content,
                metadata: $metadata,
                created_at: datetime(),
                updated_at: datetime()
            })
            RETURN d
            """,
            title=doc.title,
            content=doc.content,
            metadata=doc.metadata
        )
        return result.single()["d"]["id"]

    @staticmethod
    async def get_document(tx: Transaction, doc_id: str) -> Optional[Document]:
        result = tx.run("""
            MATCH (d:Document {id: $id})
            RETURN d
            """,
            id=doc_id
        )
        record = result.single()
        return Document(**record["d"]) if record else None

    @staticmethod
    async def update_document(tx: Transaction, doc_id: str, doc: DocumentBase) -> bool:
        result = tx.run("""
            MATCH (d:Document {id: $id})
            SET d += {
                title: $title,
                content: $content,
                metadata: $metadata,
                updated_at: datetime()
            }
            RETURN d
            """,
            id=doc_id,
            title=doc.title,
            content=doc.content,
            metadata=doc.metadata
        )
        return bool(result.single())

    @staticmethod
    async def delete_document(tx: Transaction, doc_id: str) -> bool:
        result = tx.run("""
            MATCH (d:Document {id: $id})
            DETACH DELETE d
            RETURN count(d) as deleted
            """,
            id=doc_id
        )
        return result.single()["deleted"] > 0

    @staticmethod
    async def create_entity(tx: Transaction, entity: EntityCreate) -> str:
        result = tx.run("""
            CREATE (e:Entity {
                id: randomUUID(),
                name: $name,
                type: $type,
                properties: $properties
            })
            RETURN e
            """,
            name=entity.name,
            type=entity.type,
            properties=entity.properties
        )
        return result.single()["e"]["id"]

    @staticmethod
    async def create_relationship(tx: Transaction, rel: RelationshipCreate) -> bool:
        result = tx.run("""
            MATCH (s:Entity {id: $source_id})
            MATCH (t:Entity {id: $target_id})
            CREATE (s)-[r:RELATED_TO {
                type: $type,
                properties: $properties
            }]->(t)
            RETURN r
            """,
            source_id=rel.source_id,
            target_id=rel.target_id,
            type=rel.type,
            properties=rel.properties
        )
        return bool(result.single())

# API Routes

# Document CRUD Operations
@app.post("/documents/", response_model=str)
@REQUEST_LATENCY.time()
async def create_document(document: DocumentCreate):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            doc_id = await Neo4jCRUD.create_document(session, document)
            
            # Create embeddings and store in vector index
            embeddings = ollama_emb.embed_documents([document.content])
            store = Neo4jVector.from_existing_index(
                ollama_emb,
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                index_name="vector",
                keyword_index_name="keyword",
            )
            store.add_embeddings([embeddings], [document.content], [{"doc_id": doc_id}])
            
            return doc_id
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

# Route for folder-based document creation
@app.post("/documents/from-folder/", response_model=List[str])
@REQUEST_LATENCY.time()
async def create_documents_from_folder(folder_path: str = Form(...)):
    """
    This endpoint accepts a folder path that contains PDF files. 
    The PDF files are processed, stored in Neo4j, and indexed for search.
    """
    REQUEST_COUNT.inc()

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path provided.")
    
    try:
        # Load the PDFs from the specified folder
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No PDF documents found in the specified folder.")
        
        document_ids = []
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        # Split the document content into chunks and store each one
        for doc in documents:
            split_docs = text_splitter.split_documents([doc])
            
            for split_doc in split_docs:
                # Create document in Neo4j and index it
                with driver.session() as session:
                    doc_create = DocumentCreate(
                        title=split_doc.metadata.get("title", "Untitled"),
                        content=split_doc.page_content,
                        metadata=split_doc.metadata,
                    )
                    doc_id = await Neo4jCRUD.create_document(session, doc_create)
                    
                    # Create embeddings and store in vector index
                    embeddings = ollama_emb.embed_documents([split_doc.page_content])
                    store = Neo4jVector.from_existing_index(
                        ollama_emb,
                        url=NEO4J_URI,
                        username=NEO4J_USER,
                        password=NEO4J_PASSWORD,
                        index_name="vector",
                        keyword_index_name="keyword"
                    )
                    store.add_embeddings([embeddings], [split_doc.page_content], [{"doc_id": doc_id}])
                    document_ids.append(doc_id)
        
        return document_ids
    
    except Exception as e:
        logger.error(f"Error processing documents from folder: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the folder: {e}")
    

@app.get("/documents/{doc_id}", response_model=Document)
@REQUEST_LATENCY.time()
async def get_document(doc_id: str):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            doc = await Neo4jCRUD.get_document(session, doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            return doc
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/documents/{doc_id}", response_model=bool)
@REQUEST_LATENCY.time()
async def update_document(doc_id: str, document: DocumentBase):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            updated = await Neo4jCRUD.update_document(session, doc_id, document)
            if not updated:
                raise HTTPException(status_code=404, detail="Document not found")
            return updated
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}", response_model=bool)
@REQUEST_LATENCY.time()
async def delete_document(doc_id: str):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            deleted = await Neo4jCRUD.delete_document(session, doc_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Document not found")
            return deleted
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Bulk PDF Processing
@app.post("/documents/bulk/", response_model=List[str])
@REQUEST_LATENCY.time()
async def bulk_process_pdfs(files: List[UploadFile] = File(...), folder_path: str = Form(...)):
    REQUEST_COUNT.inc()
    try:
        document_ids = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
                
            # Save temporarily and load
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Process each page as a separate document
            for doc in documents:
                doc_create = DocumentCreate(
                    title=f"{file.filename} - Page {doc.metadata.get('page', 0)}",
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                doc_id = await create_document(doc_create)
                document_ids.append(doc_id)
                
            # Cleanup
            os.remove(temp_path)
            
        return document_ids
    except Exception as e:
        logger.error(f"Error processing PDF files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Entity Operations
@app.post("/entities/", response_model=str)
@REQUEST_LATENCY.time()
async def create_entity(entity: EntityCreate):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            entity_id = await Neo4jCRUD.create_entity(session, entity)
            return entity_id
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/relationships/", response_model=bool)
@REQUEST_LATENCY.time()
async def create_relationship(relationship: RelationshipCreate):
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            created = await Neo4jCRUD.create_relationship(session, relationship)
            return created
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search and Query Operations
@app.post("/search/", response_model=List[Dict[str, Any]])
@REQUEST_LATENCY.time()
async def search_documents(query: SearchQuery):
    REQUEST_COUNT.inc()
    try:
        store = Neo4jVector.from_existing_index(
            ollama_emb,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="vector",
            keyword_index_name="keyword",
            search_type="hybrid"
        )
        
        results = store.similarity_search_with_score(
            query.query,
            k=query.limit
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata if query.include_metadata else {},
                "score": score
            }
            for doc, score in results
        ]
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Graph Query Operations
@app.post("/graph/query/", response_model=Dict[str, Any])
@REQUEST_LATENCY.time()
async def query_graph(query: str):
    REQUEST_COUNT.inc()
    try:
        qa_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True
        )
        
        result = qa_chain.invoke({"query": query})
        return {
            "cypher_query": result.get("cypher_query", ""),
            "result": result.get("result", ""),
            "intermediary_steps": result.get("intermediary_steps", [])
        }
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat/RAG Operations
@app.post("/chat/", response_model=Dict[str, str])
@REQUEST_LATENCY.time()
async def chat(query: str):
    REQUEST_COUNT.inc()
    try:
        # Setup RAG chain
        store = Neo4jVector.from_existing_index(
            ollama_emb,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="vector",
            keyword_index_name="keyword",
            search_type="hybrid"
        )
        
        retriever = store.as_retriever()
        
        prompt = PromptTemplate(
            template="""
            Based on the following context, please answer the question.
            If you cannot find the answer in the context, say "I don't have enough information to answer that."
            
            Context: {context}
            
            Question: {question}
            """,
            input_variables=["context", "question"]

)
        
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute chain
        response = await rag_chain.ainvoke(query)
        
        # Log to LangSmith
        langsmith_client.create_run(
            name="chat_rag",
            inputs={"query": query},
            outputs={"response": response},
        )
        
        return {
            "query": query,
            "response": response
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Graph Analysis Routes
@app.get("/graph/stats/", response_model=Dict[str, Any])
@REQUEST_LATENCY.time()
async def get_graph_statistics():
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            result = session.run("""
                CALL apoc.meta.stats()
                YIELD labels, relTypes, propertyKeys, nodeCount, relCount
                RETURN labels, relTypes, propertyKeys, nodeCount, relCount
            """)
            stats = result.single()
            
            return {
                "node_count": stats["nodeCount"],
                "relationship_count": stats["relCount"],
                "node_labels": stats["labels"],
                "relationship_types": stats["relTypes"],
                "property_keys": stats["propertyKeys"]
            }
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/visualization/", response_model=Dict[str, Any])
@REQUEST_LATENCY.time()
async def get_graph_visualization_data():
    REQUEST_COUNT.inc()
    try:
        with driver.session() as session:
            # Get nodes
            node_result = session.run("""
                MATCH (n)
                RETURN DISTINCT
                    id(n) as id,
                    labels(n) as labels,
                    properties(n) as properties
                LIMIT 1000
            """)
            nodes = [{"id": record["id"], "labels": record["labels"], "properties": record["properties"]}
                    for record in node_result]
            
            # Get relationships
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN DISTINCT
                    id(r) as id,
                    type(r) as type,
                    id(startNode(r)) as source,
                    id(endNode(r)) as target,
                    properties(r) as properties
                LIMIT 5000
            """)
            relationships = [{
                "id": record["id"],
                "type": record["type"],
                "source": record["source"],
                "target": record["target"],
                "properties": record["properties"]
            } for record in rel_result]
            
            return {
                "nodes": nodes,
                "relationships": relationships
            }
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector Store Management
@app.post("/vector-store/rebuild/")
@REQUEST_LATENCY.time()
async def rebuild_vector_store():
    REQUEST_COUNT.inc()
    try:
        # Get all documents
        with driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.id as id, d.content as content, d.metadata as metadata
            """)
            documents = [(record["content"], record["metadata"]) for record in result]
        
        # Create new vector store
        texts = [doc[0] for doc in documents]
        metadatas = [doc[1] for doc in documents]
        
        store = Neo4jVector.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=ollama_emb,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="vector",
            keyword_index_name="keyword",
            search_type="hybrid",
            node_label="Document",
            pre_delete_collection=True  # This will delete existing index
        )
        
        return {"message": "Vector store rebuilt successfully", "document_count": len(documents)}
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Entity Extraction
@app.post("/extract-entities/", response_model=Dict[str, List[Dict[str, Any]]])
@REQUEST_LATENCY.time()
async def extract_entities(text: str):
    REQUEST_COUNT.inc()
    try:
        # Extract entities using GLiNER
        extracted_entities = gliner_extractor.extract_nodes(text)
        
        # Transform to graph format
        graph_data = graph_transformer.transform_document(text)
        
        return {
            "entities": [
                {
                    "text": entity.text,
                    "type": entity.type,
                    "confidence": entity.score
                }
                for entity in extracted_entities
            ],
            "relationships": [
                {
                    "source": rel.source.text,
                    "target": rel.target.text,
                    "type": rel.type,
                    "confidence": rel.score
                }
                for rel in graph_data.relationships
            ] if hasattr(graph_data, 'relationships') else []
        }
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring and Health Check
@app.get("/health")
async def health_check():
    try:
        # Check Neo4j connection
        with driver.session() as session:
            session.run("RETURN 1").single()
        
        # Check Ollama embeddings
        test_embedding = ollama_emb.embed_query("test")
        
        return {
            "status": "healthy",
            "neo4j": "connected",
            "ollama": "operational",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    driver.close()
    logger.info("Application shutting down, connections closed.")

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}")
    return {
        "error": str(exc),
        "type": type(exc).__name__,
        "request_path": request.url.path
    }

if __name__ == "__main__":
    import uvicorn