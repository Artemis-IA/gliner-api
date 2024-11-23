# services/document_processing.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from pathlib import Path
from loguru import logger

from services.document_processor import DocumentProcessor
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from dependencies import get_s3_service, get_document_processor, get_mlflow_service

router = APIRouter()

# Dependency injection
s3_service: S3Service = get_s3_service()
document_processor: DocumentProcessor = get_document_processor()
mlflow_service: MLFlowService = get_mlflow_service()

@router.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[str] = Form(default=["json"]),
    use_ocr: bool = Form(False),
    export_figures: bool = Form(True),
    export_tables: bool = Form(True),
    enrich_figures: bool = Form(False)
):
    logger.info(f"Received {len(files)} files for upload")
    success_count, partial_success_count, failure_count = 0, 0, 0

    for file in files:
        temp_file = Path(f"/tmp/{file.filename}")
        with temp_file.open("wb") as out_file:
            content = await file.read()
            out_file.write(content)

        input_s3_url = s3_service.upload_file(temp_file, s3_service.input_bucket)
        document_processor.log_document(file.filename, input_s3_url)

        result = await document_processor.process_document(temp_file, use_ocr, export_figures, export_tables, enrich_figures)
        if result:
            counts = document_processor.export_document(result, export_formats, export_figures, export_tables)
            success_count += counts[0]
            partial_success_count += counts[1]
            failure_count += counts[2]

    return {
        "message": "Documents processed and stored successfully",
        "uploaded_to": s3_service.output_bucket,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count
    }

@router.post("/index_document/")
async def index_document(file: UploadFile = File(...)):
    logger.info(f"Indexing document: {file.filename}")
    temp_file = Path(f"/tmp/{file.filename}")
    with temp_file.open("wb") as out_file:
        content = await file.read()
        out_file.write(content)

    try:
        document_processor.index_document(temp_file)
        logger.info(f"Successfully indexed document: {file.filename}")
        return {"message": f"Document {file.filename} indexed successfully."}
    except Exception as e:
        logger.error(f"Error indexing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {e}")
    finally:
        temp_file.unlink()
