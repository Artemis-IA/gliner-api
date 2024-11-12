import os, time, json, yaml
from pathlib import Path
from typing import List, Tuple, Optional
import aiofiles
import pandas as pd
from enum import Enum
import torch

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem, TableItem

import mlflow
import psutil, GPUtil
from loguru import logger
from codecarbon import EmissionsTracker
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configuration des métriques Prometheus
start_http_server(8001)
REQUEST_COUNT = Counter("app_request_count", "Nombre total de requêtes")
PROCESS_TIME = Histogram("app_process_time_seconds", "Temps de traitement des requêtes")
GPU_MEMORY_USAGE = Gauge("gpu_memory_usage_bytes", "Utilisation mémoire GPU")
CPU_USAGE = Gauge("cpu_usage_percent", "Utilisation CPU")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Utilisation mémoire RAM")
CARBON_EMISSIONS = Gauge("carbon_emissions_grams", "Émissions CO2 estimées")

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5002")
mlflow.set_experiment("Document_Processing_Tracking")

# Configuration Loguru
logger.add(
    "logs/conversion_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,
    diagnose=True
)

class DeviceManager:
    def __init__(self):
        self.device = torch.device("cpu")
        self.using_gpu = torch.cuda.is_available()
        logger.info(f"Utilisation de : {self.device}")

    def log_device_stats(self):
        if self.using_gpu:
            for gpu in GPUtil.getGPUs():
                GPU_MEMORY_USAGE.set(gpu.memoryUsed)
                gpu_usage_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                logger.info(f"GPU {gpu.id} - Mémoire utilisée: {gpu.memoryUsed}MB ({gpu_usage_percent:.2f}%)")

        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info()
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory.rss)
        logger.info(f"CPU: {cpu_percent}% | Mémoire: {memory.rss / 1024 / 1024:.2f}MB")

class ModelManager:
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.emissions_tracker = EmissionsTracker(
            project_name="document_processing",
            log_level='warning'
        )
        self.tracker_active = False  # Nouveau drapeau pour l'état du tracker

    async def process_document(self, doc_converter: DocumentConverter, doc_path: Path) -> Optional[ConversionResult]:
        try:
            # Vérification de l'état du tracker et démarrage si nécessaire
            if not self.tracker_active:
                self.emissions_tracker.start()
                self.tracker_active = True

            start_time = time.time()

            with mlflow.start_run(nested=True) as run:
                mlflow.set_tag("device", str(self.device_manager.device))
                result = list(doc_converter.convert_all([doc_path], raises_on_error=False))[0]

                process_time = time.time() - start_time
                emissions = self.emissions_tracker.stop()
                self.tracker_active = False  # Remettre le drapeau à False
                CARBON_EMISSIONS.set(emissions if emissions else 0)

                # Log des métriques
                mlflow.log_metric("processing_time", process_time)
                mlflow.log_metric("carbon_emissions", emissions if emissions else 0)
                if result.status == ConversionStatus.SUCCESS:
                    mlflow.log_metric("success", 1)
                else:
                    mlflow.log_metric("failure", 1)

                self.device_manager.log_device_stats()

                logger.info(f"Document {doc_path.name} traité en {process_time:.2f}s")
                logger.info(f"Émissions CO2 estimées: {emissions:.4f}g" if emissions else "Émissions non mesurées")

                return result

        except Exception as e:
            logger.exception(f"Erreur lors du traitement de {doc_path}: {str(e)}")
            self.tracker_active = False  # Assurez-vous de remettre le drapeau à False en cas d'erreur
            return None

# Configuration FastAPI
app = FastAPI(title="Document Processing API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tables_dir = OUTPUT_DIR / "tables"
figures_dir = OUTPUT_DIR / "figures"
tables_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Initialisation des gestionnaires
device_manager = DeviceManager()
model_manager = ModelManager(device_manager)

class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

class CustomPdfPipelineOptions(PdfPipelineOptions):
    do_picture_classifier: bool = False 

def create_document_converter(use_ocr: bool, export_figures: bool, export_tables: bool, enrich_figures: bool) -> DocumentConverter:
    pipeline_options = CustomPdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = export_tables
    pipeline_options.images_scale = 2.0  # Augmente la résolution des images
    pipeline_options.generate_picture_images = export_figures
    # Désactiver temporairement `do_picture_classifier` si enrich_figures est False
    if enrich_figures:
        pipeline_options.do_picture_classifier = enrich_figures

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.HTML, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            ),
        },
    )

@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.json],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.json, ExportFormat.yaml, ExportFormat.md]
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR lors de la conversion."),
    export_figures: bool = Query(True, title="Exporter les figures", description="Activer ou désactiver l'exportation des figures"),
    export_tables: bool = Query(True, title="Exporter les tableaux", description="Activer ou désactiver l'exportation des tableaux"),
    enrich_figures: bool = Query(False, title="Enrichir les figures", description="Activer ou désactiver l'enrichissement des figures")
):
    REQUEST_COUNT.inc()
    logger.info(f"Réception de {len(files)} fichiers pour traitement")

    with mlflow.start_run(run_name="batch_processing"):
        start_time = time.time()
        input_file_paths = []
        doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)

        for file in files:
            if not file.filename.lower().endswith(('pdf', 'docx', 'pptx', 'html', 'jpg', 'jpeg', 'png')):
                raise HTTPException(
                    status_code=400,
                    detail="Format de fichier non supporté. Formats acceptés : PDF, DOCX, PPTX, HTML, JPG, JPEG, PNG"
                )

            temp_file = OUTPUT_DIR / file.filename
            async with aiofiles.open(temp_file, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            input_file_paths.append(temp_file)

        results = []
        for doc_path in input_file_paths:
            result = await model_manager.process_document(doc_converter, doc_path)
            if result:
                results.append(result)

        export_results = export_documents(results, OUTPUT_DIR, export_formats, export_figures, export_tables)

        total_time = time.time() - start_time
        PROCESS_TIME.observe(total_time)

        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("files_processed", len(results))

        return JSONResponse(content={
            "message": "Traitement terminé",
            "success_count": export_results[0],
            "partial_success_count": export_results[1],
            "failure_count": export_results[2],
            "processing_time": f"{total_time:.2f}s"
        })

@app.post("/upload_path/")
async def upload_path(
    file_path: str = Form(..., title="Chemin du dossier", description="Chemin vers le dossier contenant les fichiers"),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.json],
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation"
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR pour la conversion."),
    export_figures: bool = Query(True, title="Exporter les figures", description="Activer ou désactiver l'exportation des figures"),
    export_tables: bool = Query(True, title="Exporter les tableaux", description="Activer ou désactiver l'exportation des tableaux"),
    enrich_figures: bool = Query(False, title="Enrichir les figures", description="Activer ou désactiver l'enrichissement des figures")
):
    REQUEST_COUNT.inc()
    logger.info(f"Traitement du dossier: {file_path}")
    input_dir_path = Path(file_path)

    if not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Le chemin spécifié n'est pas un répertoire valide.")

    # Extensions de fichiers supportées
    supported_extensions = ['.pdf', '.docx', '.pptx', '.html', '.jpg', '.jpeg', '.png']

    # Filtrer les fichiers avec les extensions supportées
    input_file_paths = [
        file for file in input_dir_path.glob('*')
        if file.is_file() and file.suffix.lower() in supported_extensions
    ]

    if not input_file_paths:
        raise HTTPException(status_code=400, detail="Aucun fichier valide dans le répertoire.")

    with mlflow.start_run(run_name="directory_processing"):
        start_time = time.time()
        doc_converter = create_document_converter(use_ocr, export_figures, export_tables, enrich_figures)

        results = []
        for doc_path in input_file_paths:
            result = await model_manager.process_document(doc_converter, doc_path)
            if result:
                results.append(result)

        export_results = export_documents(results, OUTPUT_DIR, export_formats, export_figures, export_tables)

        total_time = time.time() - start_time
        PROCESS_TIME.observe(total_time)

        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("files_processed", len(results))

        return JSONResponse(content={
            "success_count": export_results[0],
            "partial_success_count": export_results[1],
            "failure_count": export_results[2],
            "total_processed": len(results),
            "processing_time": f"{total_time:.2f}s"
        })

def export_documents(conv_results: List[ConversionResult], output_dir: Path, export_formats: List[ExportFormat], export_figures: bool, export_tables: bool) -> Tuple[int, int, int]:
    success_count = 0
    partial_success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1

            # Log additionnel pour vérifier le nombre d'images détectées
            num_figures = len([e for e in conv_res.document.iterate_items() if isinstance(e, PictureItem)])
            logger.info(f"Détection de {num_figures} figures dans le document {doc_filename}.")

            # Export des résultats dans les formats sélectionnés
            if ExportFormat.json in export_formats:
                json_path = output_dir / f"{doc_filename}.json"
                with json_path.open("w", encoding='utf-8') as json_file:
                    json.dump(conv_res.document.export_to_dict(), json_file, ensure_ascii=False, indent=2)

            if ExportFormat.yaml in export_formats:
                yaml_path = output_dir / f"{doc_filename}.yaml"
                with yaml_path.open("w", encoding='utf-8') as yaml_file:
                    yaml.dump(
                        conv_res.document.export_to_dict(),
                        yaml_file,
                        allow_unicode=True,
                        default_flow_style=False
                    )

            if ExportFormat.md in export_formats:
                md_path = output_dir / f"{doc_filename}.md"
                with md_path.open("w", encoding='utf-8') as md_file:
                    md_file.write(conv_res.document.export_to_markdown())

            # Exportation des figures (images)
            figures_dir = OUTPUT_DIR / "figures"
            figures_dir.mkdir(exist_ok=True)

            if export_figures:
                for idx, element in enumerate(conv_res.document.iterate_items()):
                    if isinstance(element, PictureItem):
                        figure_path = figures_dir / f"{doc_filename}_figure_{idx + 1}.png"
                        element.image.pil_image.save(figure_path, format="PNG")
                        logger.info(f"Figure exportée: {figure_path}")

            # Exportation des tableaux
            tables_dir = OUTPUT_DIR / "tables"
            tables_dir.mkdir(exist_ok=True)

            if export_tables:
                for table_idx, table in enumerate(conv_res.document.tables):
                    table_df = table.export_to_dataframe()

                    # Export CSV
                    csv_path = tables_dir / f"{doc_filename}_table_{table_idx + 1}.csv"
                    table_df.to_csv(csv_path, index=False, encoding='utf-8')

                    # Export HTML
                    html_path = tables_dir / f"{doc_filename}_table_{table_idx + 1}.html"
                    with html_path.open("w", encoding='utf-8') as html_file:
                        html_file.write(table.export_to_html())

                    logger.info(f"Table exportée: {csv_path} et {html_path}")

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.info(f"Document {conv_res.input.file} partiellement converti")
            partial_success_count += 1
        else:
            logger.error(f"Échec de la conversion pour {conv_res.input.file}")
            failure_count += 1

    logger.info(f"Documents traités: {len(conv_results)} "
                f"(Succès: {success_count}, "
                f"Partiels: {partial_success_count}, "
                f"Échecs: {failure_count})")
    return success_count, partial_success_count, failure_count

@app.get("/")
def read_root():
    return {
        "message": "API de traitement de documents v2.0",
        "status": "active",
        "device": str(device_manager.device)
    }

@app.get("/files/")
async def get_converted_files():
    if not OUTPUT_DIR.exists():
        return JSONResponse(
            content={"message": "Aucun fichier trouvé."},
            status_code=404
        )

    converted_files = []
    for file_path in OUTPUT_DIR.iterdir():
        if file_path.is_file():
            converted_files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })

    if not converted_files:
        return JSONResponse(content={"message": "Aucun fichier converti."}, status_code=404)

    return JSONResponse(content={"converted_files": converted_files})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
