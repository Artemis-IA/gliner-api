# dependencies.py
from fastapi import Depends
from sqlalchemy.orm import Session
from utils.database import SessionLocal
from config import settings
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from services.document_processor import DocumentProcessor
from services.neo4j_service import Neo4jService

# Dependency to get the SQLAlchemy session
from typing import Generator

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Example of other shared dependencies
async def common_parameters(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Dependency to get the S3 service
def get_s3_service() -> S3Service:
    return S3Service(
        s3_client=None,
        endpoint_url=settings.MINIO_URL,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        input_bucket="docs-input",
        output_bucket="docs-output",
        layouts_bucket="layouts"
    )


# Dependency to get the MLflow service
def get_mlflow_service() -> MLFlowService:
    return MLFlowService(tracking_uri=settings.MLFLOW_TRACKING_URI)

# Dependency to get the document processor
def get_document_processor(db: Session = Depends(get_db)) -> DocumentProcessor:
    s3_service = get_s3_service()
    mlflow_service = get_mlflow_service()
    return DocumentProcessor(
        s3_service=s3_service,
        mlflow_service=mlflow_service,
        session=db
    )

# Dependency to get the Neo4j service
def get_neo4j_service() -> Neo4jService:
    return Neo4jService(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )