import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, Query, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import aiofiles
import boto3
import json
import yaml
from loguru import logger

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem

import torch
import psutil
import GPUtil

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Initialisation de l'application
app = FastAPI(title="Document Processing API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des répertoires locaux
LOCAL_INPUT_DIR = Path("local_input")
LOCAL_OUTPUT_DIR = Path("local_output")
LOCAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration S3
MINIO_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
input_bucket = "docs-input"
output_bucket = "docs-output"
layouts_bucket = "layouts"

# Configuration SQLAlchemy
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Configuration du client S3
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)
for bucket in [input_bucket, output_bucket, layouts_bucket]:
    try:
        s3_client.head_bucket(Bucket=bucket)
    except:
        s3_client.create_bucket(Bucket=bucket)

# Enumération des formats d'import et d'export
class ImportFormat(str, Enum):
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"

class ExportFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "md"
    DOCTAGS = "doctags"

# Définition de CustomPdfPipelineOptions
class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False 

# Modèle de log SQLAlchemy
class DocumentLog(Base):
    __tablename__ = "document_logs"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    s3_url = Column(String)

Base.metadata.create_all(bind=engine)

class DocumentLogService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def log_document(self, file_name: str, s3_url: Optional[str] = None):
        with self.session_factory() as session:
            log = DocumentLog(file_name=file_name, s3_url=s3_url)
            session.add(log)
            session.commit()

class S3Service:
    def __init__(self, client, input_bucket, output_bucket, layouts_bucket):
        self.client = client
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.layouts_bucket = layouts_bucket

    def upload_file(self, file_path: Path, bucket_name: str):
        try:
            self.client.upload_file(str(file_path), bucket_name, file_path.name)
            logger.info(f"Uploaded {file_path.name} to S3 bucket '{bucket_name}'")
            return f"s3://{bucket_name}/{file_path.name}"
        except Exception as e:
            logger.error(f"Failed to upload {file_path.name} to {bucket_name}: {e}")
            return None

    def download_file(self, s3_url: str, local_path: Path):
        try:
            bucket_name, key = self._parse_s3_url(s3_url)
            self.client.download_file(bucket_name, key, str(local_path))
            logger.info(f"Downloaded {s3_url} to local path '{local_path}'")
        except Exception as e:
            logger.error(f"Failed to download {s3_url}: {e}")

    @staticmethod
    def _parse_s3_url(s3_url: str):
        bucket_name = s3_url.split("//")[1].split("/")[0]
        key = "/".join(s3_url.split("//")[1].split("/")[1:])
        return bucket_name, key

class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.using_gpu = self.device.type == "cuda"
        logger.info(f"Using device: {self.device}")

    def log_device_stats(self):
        if self.using_gpu:
            for gpu in GPUtil.getGPUs():
                gpu_usage_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                logger.info(f"GPU {gpu.id} - Memory used: {gpu.memoryUsed}MB ({gpu_usage_percent:.2f}%)")
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info()
        logger.info(f"CPU: {cpu_percent}% | Memory: {memory.rss / 1024 / 1024:.2f}MB")

# Prometheus metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests")
TABLE_COUNT = Counter("app_table_count", "Total number of tables processed")
FIGURE_COUNT = Counter("app_figure_count", "Total number of figures processed")
START_TIME = Histogram('app_start_time', 'Time when the request was started')
PROCESS_TIME = Histogram('app_process_time_seconds', 'Time spent processing request')
start_http_server(8000)  # Start Prometheus server on port 8000

class DocumentProcessor:
    def __init__(self, s3_service: S3Service, session_factory):
        self.s3_service = s3_service
        self.doc_log_service = DocumentLogService(session_factory)
        self.device_manager = DeviceManager()

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool):
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.IMAGE, InputFormat.HTML],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)},
        )

    async def process_document(self, converter, doc_path: Path, model_name: str, source: str):
        logger.info(f"Processing document '{doc_path.name}' from {source}")
        self.device_manager.log_device_stats()
        result = list(converter.convert_all([doc_path]))[0]
        return result

    def export_document(self, result, output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool, destination: str):
        success_count, partial_success_count, failure_count = 0, 0, 0
        doc_filename = result.input.file.stem

        if result.status == ConversionStatus.SUCCESS:
            success_count += 1
            logger.info(f"Document '{doc_filename}' converted successfully")
            self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename, destination)
        elif result.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
            logger.warning(f"Document '{doc_filename}' converted with partial success")
        else:
            failure_count += 1
            logger.error(f"Document '{doc_filename}' failed to convert")

        return success_count, partial_success_count, failure_count

    def _export_file(self, result, output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool, doc_filename: str, destination: str):
        if ExportFormat.JSON in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "json", export_format="json", destination=destination)
        if ExportFormat.YAML in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "yaml", export_format="yaml", destination=destination)
        if ExportFormat.MARKDOWN in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "md", export_format="md", destination=destination)
        if ExportFormat.TEXT in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "txt", export_format="md", destination=destination)
        if ExportFormat.DOCTAGS in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "doctags", export_format="json", destination=destination)

        if export_figures:
            figure_count = self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket, destination)
            FIGURE_COUNT.inc(figure_count)
            logger.info(f"Exported {figure_count} figures from document '{doc_filename}' to {destination}")
        if export_tables:
            table_count = self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket, destination)
            TABLE_COUNT.inc(table_count)
            logger.info(f"Exported {table_count} tables from document '{doc_filename}' to {destination}")

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json", destination="local"):
        file_path = output_dir / f"{doc_filename}.{ext}"
        with file_path.open("w", encoding="utf-8") as file:
            if export_format == "json":
                json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
            elif export_format == "yaml":
                yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
            elif export_format == "md":
                file.write(result.document.export_to_markdown())
        if destination == "s3":
            self.s3_service.upload_file(file_path, self.s3_service.output_bucket)
            logger.info(f"Exported '{file_path.name}' to S3 bucket '{self.s3_service.output_bucket}'")
        else:
            logger.info(f"Exported '{file_path.name}' to local path '{file_path}'")

    def _export_images(self, result, figures_dir, doc_filename, bucket, destination):
        figures_dir.mkdir(exist_ok=True)
        count = 0
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                count += 1
                if destination == "s3":
                    self.s3_service.upload_file(image_path, bucket)
                else:
                    logger.info(f"Saved figure '{image_path.name}' to local path '{image_path}'")
        return count

    def _export_tables(self, result, tables_dir, doc_filename, bucket, destination):
        tables_dir.mkdir(exist_ok=True)
        count = 0
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            count += 1
            if destination == "s3":
                self.s3_service.upload_file(csv_path, bucket)
            else:
                logger.info(f"Saved table '{csv_path.name}' to local path '{csv_path}'")
        return count

# Initialisation des services
s3_service = S3Service(s3_client, input_bucket, output_bucket, layouts_bucket)
processor = DocumentProcessor(s3_service, SessionLocal)

@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[ExportFormat] = Query(default=[ExportFormat.JSON]),
    use_s3: bool = Query(False),
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False
):
    REQUEST_COUNT.inc()
    source = "S3" if use_s3 else "local filesystem"
    destination = "S3" if use_s3 else "local filesystem"
    logger.info(f"Received {len(files)} files for processing from {source}.")
    output_dir = LOCAL_OUTPUT_DIR if not use_s3 else Path("/tmp")

    converter = processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)
    success_count, partial_success_count, failure_count = 0, 0, 0

    for file in files:
        temp_file = (LOCAL_INPUT_DIR if not use_s3 else output_dir) / file.filename
        async with aiofiles.open(temp_file, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        if use_s3:
            input_s3_url = s3_service.upload_file(temp_file, input_bucket)
            processor.doc_log_service.log_document(file.filename, input_s3_url)
            logger.info(f"Uploaded '{file.filename}' to S3 bucket '{input_bucket}'")
        else:
            processor.doc_log_service.log_document(file.filename)
            logger.info(f"Saved '{file.filename}' to local path '{temp_file}'")

        result = await processor.process_document(converter, temp_file, model_name="Docling", source=source)
        if result:
            counts = processor.export_document(result, output_dir, export_formats, export_figures, export_tables, destination=destination)
            success_count += counts[0]
            partial_success_count += counts[1]
            failure_count += counts[2]

    return {
        "message": "Documents processed and stored successfully",
        "uploaded_to": destination,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count
    }

@app.post("/upload_path/")
async def upload_path(
    file_path: str = Form(...),
    export_formats: List[ExportFormat] = Query(default=[ExportFormat.JSON]),
    use_s3: bool = Query(False),
    use_ocr: bool = False,
    export_figures: bool = True,
    export_tables: bool = True,
    enrich_figures: bool = False
):
    REQUEST_COUNT.inc()
    source = "S3" if use_s3 else "local filesystem"
    destination = "S3" if use_s3 else "local filesystem"
    logger.info(f"Processing directory '{file_path}' from {source}.")
    output_dir = LOCAL_OUTPUT_DIR if not use_s3 else Path("/tmp")
    converter = processor.create_converter(use_ocr, export_figures, export_tables, enrich_figures)
    success_count, partial_success_count, failure_count = 0, 0, 0

    if use_s3:
        objects = s3_client.list_objects_v2(Bucket=input_bucket, Prefix=file_path).get("Contents", [])
        logger.info(f"Found {len(objects)} files in S3 bucket '{input_bucket}' with prefix '{file_path}'")
        for obj in objects:
            temp_file = Path("/tmp") / obj["Key"].split("/")[-1]
            s3_service.download_file(f"s3://{input_bucket}/{obj['Key']}", temp_file)
            processor.doc_log_service.log_document(temp_file.name)
            result = await processor.process_document(converter, temp_file, model_name="Docling", source=source)
            if result:
                counts = processor.export_document(result, output_dir, export_formats, export_figures, export_tables, destination=destination)
                success_count += counts[0]
                partial_success_count += counts[1]
                failure_count += counts[2]
    else:
        input_dir = Path(file_path)
        files = list(input_dir.glob("*"))
        logger.info(f"Found {len(files)} files in local directory '{file_path}'")
        for doc_path in files:
            if doc_path.suffix.lower() in [".pdf", ".docx"]:
                processor.doc_log_service.log_document(doc_path.name)
                result = await processor.process_document(converter, doc_path, model_name="Docling", source=source)
                if result:
                    counts = processor.export_document(result, output_dir, export_formats, export_figures, export_tables, destination=destination)
                    success_count += counts[0]
                    partial_success_count += counts[1]
                    failure_count += counts[2]

    return {
        "message": "Directory processed and stored successfully",
        "uploaded_to": destination,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count
    }
