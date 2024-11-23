# services/document_processor.py
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from services.s3_service import S3Service
from services.mlflow_service import MLFlowService
from sqlalchemy.orm import Session
from models.sqlalchemy.document_log import DocumentLog

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False

class DocumentProcessor:
    def __init__(self, s3_service: S3Service, mlflow_service: MLFlowService, session: Session):
        self.s3_service = s3_service
        self.mlflow_service = mlflow_service
        self.session = session or (session_factory and session_factory())

    def create_converter(self, use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool):
        options = CustomPdfPipelineOptions()
        options.do_ocr = use_ocr
        options.generate_page_images = True
        options.generate_table_images = export_tables
        options.generate_picture_images = export_figures
        options.do_picture_classifier = enrich_figures
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)}
        )

    def log_document(self, file_name: str, s3_url: str):
        log = DocumentLog(file_name=file_name, s3_url=s3_url)
        self.session.add(log)
        self.session.commit()

    def export_document(self, result: ConversionResult, output_dir: Path, export_formats: List[str], export_figures: bool, export_tables: bool):
        success_count, partial_success_count, failure_count = 0, 0, 0
        doc_filename = result.input.file.stem

        if result.status == ConversionStatus.SUCCESS:
            success_count += 1
            self._export_file(result, output_dir, export_formats, export_figures, export_tables, doc_filename)
        elif result.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
        else:
            failure_count += 1

        return success_count, partial_success_count, failure_count

    def _export_file(self, result, output_dir, export_formats: List[str], export_figures: bool, export_tables: bool, doc_filename: str):
        if "json" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "json", export_format="json")
        if "yaml" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "yaml", export_format="yaml")
        if "md" in export_formats:
            self._save_and_upload(result, output_dir, doc_filename, "md", export_format="md")

        if export_figures:
            self._export_images(result, output_dir / "figures", doc_filename, self.s3_service.layouts_bucket)
        if export_tables:
            self._export_tables(result, output_dir / "tables", doc_filename, self.s3_service.layouts_bucket)

    def _save_and_upload(self, result, output_dir, doc_filename, ext, export_format="json"):
        file_path = output_dir / f"{doc_filename}.{ext}"
        with file_path.open("w", encoding="utf-8") as file:
            if export_format == "json":
                json.dump(result.document.export_to_dict(), file, ensure_ascii=False, indent=2)
            elif export_format == "yaml":
                yaml.dump(result.document.export_to_dict(), file, allow_unicode=True)
            elif export_format == "md":
                file.write(result.document.export_to_markdown())
        self.s3_service.upload_file(file_path, self.s3_service.output_bucket)

    def _export_images(self, result, figures_dir, doc_filename, bucket):
        figures_dir.mkdir(exist_ok=True)
        for idx, element in enumerate(result.document.iterate_items()):
            if isinstance(element, PictureItem):
                image_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                element.image.pil_image.save(image_path, format="PNG")
                self.s3_service.upload_file(image_path, bucket)

    def _export_tables(self, result, tables_dir, doc_filename, bucket):
        tables_dir.mkdir(exist_ok=True)
        for idx, table in enumerate(result.document.tables):
            csv_path = tables_dir / f"{doc_filename}_table_{idx + 1}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False, encoding="utf-8")
            self.s3_service.upload_file(csv_path, bucket)

            html_path = tables_dir / f"{doc_filename}_table_{idx + 1}.html"
            with html_path.open("w", encoding="utf-8") as html_file:
                html_file.write(table.export_to_html())
            self.s3_service.upload_file(html_path, bucket)
