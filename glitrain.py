import os
import json
import yaml
import fitz  # PyMuPDF
import torch
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3
from loguru import logger
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.data_processing import GLiNERDataset

# Initialisation de l'application FastAPI
app = FastAPI(title="GLiNER PDF Processing API", description="API for processing PDFs and storing datasets", version="1.0")

# Configurations S3
s3_client = boto3.client(
    's3',
    endpoint_url='http://127.0.0.1:9000',  # URL de votre service MinIO ou S3
    aws_access_key_id='new_access_key',
    aws_secret_access_key='new_secret_key'
)
bucket_name = 'pdfs'
try:
    s3_client.head_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' already exists.")
except:
    s3_client.create_bucket(Bucket=bucket_name)
    logger.info(f"Bucket '{bucket_name}' created.")

# Configurations PostgreSQL
DATABASE_URL = "postgresql://postgre_user:postgre_password@localhost/postgre_db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class DatasetLog(Base):
    __tablename__ = 'dataset_logs'
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    s3_url = Column(String)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# Liste des modèles disponibles
AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Classe Pydantic pour les autres données
class PDFInput(BaseModel):
    filepath: str = Field(..., description="Chemin vers le fichier ou le répertoire PDF à traiter")
    gliner_model: str = Field(..., description="Sélectionnez un modèle GLiNER à utiliser", enum=AVAILABLE_MODELS)

class ProcessedOutput(BaseModel):
    s3_path: str = Field(..., example="s3://gliner_datasets/processed_data.json")

# Fonctions auxiliaires
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF en utilisant PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def load_labels(labels_file: UploadFile) -> List[str]:
    """Charge les labels à partir d'un fichier YAML ou JSON."""
    content = labels_file.file.read().decode()
    if labels_file.filename.endswith(".yml") or labels_file.filename.endswith(".yaml"):
        labels_data = yaml.safe_load(content)
    elif labels_file.filename.endswith(".json"):
        labels_data = json.loads(content)
    else:
        raise HTTPException(status_code=400, detail="Le fichier de labels doit être au format .yml, .yaml ou .json")
    return labels_data.get("labels", [])

def annotate_text(model, text: str, labels: List[str]) -> List[Dict[str, Union[str, int, float]]]:
    """Annoter le texte avec les entités nommées en utilisant GLiNER."""
    labels = [label.strip() for label in labels]
    entities = model.predict_entities(text, labels)
    return [
        {
            "entity": entity["label"],
            "word": entity["text"],
            "start": entity["start"],
            "end": entity["end"],
            "score": entity.get("score", 0)
        }
        for entity in entities
    ]

def save_to_s3(file_path: str, bucket: str) -> str:
    """Envoie le fichier spécifié vers S3 et retourne l'URL."""
    s3_key = os.path.basename(file_path)
    s3_client.upload_file(file_path, bucket, s3_key)
    s3_url = f"s3://{bucket}/{s3_key}"
    return s3_url

def log_to_postgres(file_name: str, s3_url: str):
    """Enregistre les informations dans PostgreSQL."""
    session = SessionLocal()
    dataset_log = DatasetLog(file_name=file_name, s3_url=s3_url)
    session.add(dataset_log)
    session.commit()
    session.close()

# Endpoint unique pour traiter le PDF ou le répertoire
@app.post("/process_pdf/", response_model=ProcessedOutput)
async def process_pdf(
    filepath: str = Form("/home/pi/Documents/IF-SRV/4pdfs_subset/", description="Chemin vers le fichier ou le répertoire PDF à traiter"),
    gliner_model: str = Form(..., description="Sélectionnez un modèle GLiNER à utiliser", enum=AVAILABLE_MODELS),
    labels_file: Optional[UploadFile] = File(None, description="Fichier .yml ou .json contenant les labels")
):
    """
    Processus complet pour transformer un fichier PDF en jeu de données GLiNER-compatible, et stocker dans S3.
    """
    # Vérifier que le chemin existe
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Fichier ou répertoire non trouvé.")

    # Charger les labels
    if labels_file:
        labels = load_labels(labels_file)
    else:
        labels = ["person", "organization", "location"]  # Valeurs par défaut

    # Charger le modèle GLiNER pour l'annotation
    if gliner_model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Modèle non trouvé.")
    
    model = GLiNER.from_pretrained(gliner_model).to(device)

    dataset_entries = []

    # Traitement unique ou multiple selon si filepath est un fichier ou un répertoire
    if os.path.isfile(filepath):
        text = extract_text_from_pdf(filepath)
        annotations = annotate_text(model, text, labels)
        dataset_entries.append({"text": text, "entities": annotations})

    elif os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                    annotations = annotate_text(model, text, labels)
                    dataset_entries.append({"text": text, "entities": annotations})
    else:
        raise HTTPException(status_code=400, detail="Le chemin spécifié n'est ni un fichier ni un répertoire.")

    # Sauvegarder les données annotées dans un fichier JSON
    dataset_filename = f"{os.path.basename(filepath)}_dataset.json"
    dataset_path = os.path.join("data", dataset_filename)
    with open(dataset_path, "w", encoding="utf-8") as dataset_file:
        json.dump(dataset_entries, dataset_file, ensure_ascii=False, indent=4)

    # Stocker dans S3
    s3_url = save_to_s3(dataset_path, bucket_name)

    # Log dans PostgreSQL
    log_to_postgres(dataset_filename, s3_url)

    return {"s3_path": s3_url}
