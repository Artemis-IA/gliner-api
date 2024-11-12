import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import aiofiles
import pandas as pd
from enum import Enum

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem, TableItem

# Configuration de logging
logging.basicConfig(level=logging.INFO)

# Définition de l'application FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À changer en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enum pour les formats d'exportation
class ExportFormat(str, Enum):
    json = "json"
    yaml = "yaml"
    md = "md"

# Fonction pour créer un DocumentConverter
def create_document_converter(use_ocr: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr  # Changer en True ou False selon l'argument

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.HTML,
            InputFormat.IMAGE,
        ],
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
        default=[ExportFormat.json],  # Valeur par défaut
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.json, ExportFormat.yaml, ExportFormat.md]
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR lors de la conversion.")
):
    """
    Endpoint pour télécharger plusieurs fichiers et les traiter.
    """
    input_file_paths = []
    doc_converter = create_document_converter(use_ocr)  # Création du DocumentConverter avec OCR

    for file in files:
        if not file.filename.endswith(('pdf', 'docx', 'pptx', 'html')):
            raise HTTPException(status_code=400, detail="Tous les fichiers doivent être des PDF, DOCX, PPTX ou HTML.")

        # Enregistrement temporaire des fichiers
        temp_file = OUTPUT_DIR / file.filename
        async with aiofiles.open(temp_file, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        input_file_paths.append(temp_file)

    start_time = time.time()

    # Conversion des documents
    conv_results = list(doc_converter.convert_all(input_file_paths, raises_on_error=False))
    
    export_results = export_documents(conv_results, OUTPUT_DIR, export_formats)

    end_time = time.time() - start_time
    logging.info(f"Conversion terminée en {end_time:.2f} secondes.")

    return JSONResponse(content={
        "success_count": export_results[0],
        "partial_success_count": export_results[1],
        "failure_count": export_results[2],
        "total_processed": len(input_file_paths),
    })

@app.post("/upload_path/")
async def upload_path(
    file_path: str = Form("/home/Ibrahim.Mohammad/Documents/dataset-cac40-pdf-subset/", title="Chemin du dossier", description="Chemin vers le dossier contenant les fichiers à traiter"),
    export_formats: List[ExportFormat] = Query(
        default=[ExportFormat.json],  # Valeur par défaut
        title="Formats d'exportation",
        description="Choisissez les formats d'exportation",
        enum=[ExportFormat.json, ExportFormat.yaml, ExportFormat.md]
    ),
    use_ocr: bool = Query(False, title="Utiliser OCR", description="Activez l'OCR lors de la conversion.")
):
    """
    Endpoint pour traiter tous les fichiers dans un dossier spécifié.
    """
    input_dir_path = Path(file_path)
    
    if not input_dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Le chemin spécifié n'est pas un répertoire valide.")

    # Récupérer tous les fichiers du répertoire
    input_file_paths = list(input_dir_path.glob('*'))  # Tous les fichiers dans le répertoire

    # Filtrer pour ne garder que ceux qui sont des fichiers
    input_file_paths = [file for file in input_file_paths if file.is_file() and file.suffix in ['.pdf', '.docx', '.pptx', '.html']]

    if not input_file_paths:
        raise HTTPException(status_code=400, detail="Aucun fichier valide dans le répertoire.")

    start_time = time.time()
    
    doc_converter = create_document_converter(use_ocr)  # Création du DocumentConverter avec OCR

    # Conversion des documents
    conv_results = list(doc_converter.convert_all(input_file_paths, raises_on_error=False))
    
    export_results = export_documents(conv_results, OUTPUT_DIR, export_formats)

    end_time = time.time() - start_time
    logging.info(f"Conversion terminée en {end_time:.2f} secondes.")

    return JSONResponse(content={
        "success_count": export_results[0],
        "partial_success_count": export_results[1],
        "failure_count": export_results[2],
        "total_processed": len(conv_results),
    })

def export_documents(conv_results: List[ConversionResult], output_dir: Path, export_formats: List[ExportFormat]) -> Tuple[int, int, int]:
    """
    Fonction pour exporter les résultats de conversion dans les formats spécifiés, y compris les figures et les tableaux.
    """
    success_count = 0
    partial_success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            
            # Export des résultats dans les formats sélectionnés
            if ExportFormat.json in export_formats:
                with (output_dir / f"{doc_filename}.json").open("w") as json_file:
                    json_file.write(json.dumps(conv_res.document.export_to_dict()))

            if ExportFormat.yaml in export_formats:
                with (output_dir / f"{doc_filename}.yaml").open("w") as yaml_file:
                    yaml_file.write(yaml.dump(conv_res.document.export_to_dict()))

            if ExportFormat.md in export_formats:
                with (output_dir / f"{doc_filename}.md").open("w") as md_file:
                    md_file.write(conv_res.document.export_to_markdown())

            # Exportation des figures
            figure_counter = 0
            for element in conv_res.document.iterate_items():
                if isinstance(element, PictureItem):
                    figure_counter += 1
                    figure_filename = output_dir / f"{doc_filename}-figure-{figure_counter}.png"
                    with figure_filename.open("wb") as fig_file:
                        element.image.pil_image.save(fig_file, format="PNG")

            # Exportation des tableaux
            for table_ix, table in enumerate(conv_res.document.tables):
                table_df: pd.DataFrame = table.export_to_dataframe()
                
                # Exporter en CSV
                element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
                table_df.to_csv(element_csv_filename, index=False)

                # Exporter en HTML
                element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
                with element_html_filename.open("w") as html_file:
                    html_file.write(table.export_to_html())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            logging.info(f"Document {conv_res.input.file} a été partiellement converti.")
            partial_success_count += 1
        else:
            logging.error(f"Échec de la conversion pour le document {conv_res.input.file}.")
            failure_count += 1

    logging.info(f"Documents traités : {len(conv_results)} (Succès : {success_count}, Partiels : {partial_success_count}, Échecs : {failure_count}).")
    return success_count, partial_success_count, failure_count

@app.get("/")
def read_root():
    return {"message": "API fonctionne. Utilisez l'endpoint /upload pour télécharger des fichiers."}

@app.get("/files/")
async def get_converted_files():
    """
    Endpoint pour obtenir la liste des fichiers convertis.
    """
    if not OUTPUT_DIR.exists():
        return JSONResponse(content={"message": "Aucun fichier trouvé."}, status_code=404)

    # Liste des fichiers dans le dossier de sortie
    converted_files = [file.name for file in OUTPUT_DIR.iterdir() if file.is_file()]

    if not converted_files:
        return JSONResponse(content={"message": "Aucun fichier converti."}, status_code=404)

    return JSONResponse(content={"converted_files": converted_files})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)