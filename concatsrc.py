
# miniobucket.py
#-----
# from gliner import GLiNER
# import json

# model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
# config = model.config.to_dict()
# print(json.dumps(config, indent=4))


from minio import S3Error
from minio import Minio

# Initialize Minio client
minio_client = Minio(
    "play.min.io",
    access_key="minio",
    secret_key="minio123",
    secure=True
)

try:
    bucket_name = "datasets"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
except S3Error as e:
    print(f"MinIO Error: {e}")

#-----

# main.py
#-----
# src/main.py

from fastapi import FastAPI
from routers import auth, dataset, inference, train, ml_backend
from utils.mlflow_manager import MLflowManager
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from db.init_db import init_db
import uvicorn
import logging

app = FastAPI(
    title="GLiNER CRUD API",
    description="API",
    version="1.0.0"
)

mlflow_manager = MLflowManager()

@app.on_event("startup")
def on_startup():
    logging.info("Démarrage de l'application...")
    # init_db()
    # mlflow_manager.setup_mlflow()
    # mlflow_manager.run_migrations()
    # ml_backend.start()


@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).observe(process_time)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()

    return response

app.include_router(auth.router, prefix="/auth")
app.include_router(dataset.router, prefix="/dataset")
app.include_router(inference.router, prefix="/inference")
app.include_router(train.router, prefix="/train")
app.include_router(ml_backend.router, prefix="/ml")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajouter l'endpoint Prometheus
app.mount("/metrics", make_asgi_app())

# Point de terminaison racine
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API GLiNER CRUD"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8008, reload=True)

#-----

# core/config.py
#-----
# src/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    # Variables d'environnement
    database_url: str
    mlflow_tracking_uri: str
    mlflow_backend_store_uri: str
    mlflow_artifact_root: str
    mlflow_db: str
    mlflow_port: int

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_port: int
    postgres_host: str
    
    minio_port: int
    minio_root_user: str
    minio_root_password: str
    minio_api_url: str
    
    mlflow_s3_endpoint_url: str
    mlflow_s3_ignore_tls: bool
    
    prometheus_port: int = 8008
    
    default_models: str = "urchade/gliner_smallv2.1"

    train_config: dict = {
        "num_steps": 10_000,  # N training iteration
        "train_batch_size": 2,  # batch size for training
        "eval_every": 1_000,  # evaluation/saving steps
        "save_directory": "checkpoints",  # where to save checkpoints
        "warmup_ratio": 0.1,  # warmup steps
        "device": "cuda",  # placeholder, will be set dynamically
        "lr_encoder": 1e-5,  # learning rate for the backbone
        "lr_others": 5e-5,  # learning rate for other parameters
        "freeze_token_rep": False,  # freeze of not the backbone
        "max_types": 25,  # maximum number of entity types during training
        "shuffle_types": True,  # if shuffle or not entity types
        "random_drop": True,  # randomly drop entity types
        "max_neg_type_ratio": 1,  # ratio of positive/negative types
        "max_len": 384,  # maximum sentence length
    }

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = 'utf-8'

# Instanciation de la configuration
settings = Settings()

MODELS = {
    "GLiNER-S": "urchade/gliner_smallv2.1",
    "GLiNER-M": "urchade/gliner_mediumv2.1",
    "GLiNER-L": "urchade/gliner_largev2.1",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CONFIG = {
    "num_steps": 10_000,
    "train_batch_size": 2,
    "eval_every": 1_000,
    "save_directory": "checkpoints",
    "warmup_ratio": 0.1,
    "device": DEVICE,
    "lr_encoder": 1e-5,
    "lr_others": 5e-5,
    "freeze_token_rep": False,
    "max_types": 25,
    "shuffle_types": True,
    "random_drop": True,
    "max_neg_type_ratio": 1,
    "max_len": 384,
}
#-----

# core/__init__.py
#-----

#-----

# schemas/__init__.py
#-----

#-----

# schemas/dataset.py
#-----
# src/schemas/dataset.py
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class DatasetCreate(BaseModel):
    name: str = Field(..., example="Sample Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Example", "entities": ["Entity1", "Entity2"]}])

class DatasetUpdate(BaseModel):
    name: str = Field(..., example="Updated Dataset")
    data: List[Dict] = Field(..., example=[{"text": "Updated Example", "entities": ["Entity3"]}])

class DatasetResponse(BaseModel):
    id: int
    name: str
    data: List[Dict]
    created_at: datetime

    class Config:
        from_attributes = True


#-----

# schemas/models_dict.py
#-----
# src/schemas/models_dict.py

from enum import Enum

class ModelName(str, Enum):
    GLiNER_S = "GLiNER-S"
    GLiNER_M = "GLiNER-M"
    GLiNER_L = "GLiNER-L"
    GLiNER_News = "GLiNER-News"
    GLiNER_PII = "GLiNER-PII"
    GLiNER_Bio = "GLiNER-Bio"
    GLiNER_Bird = "GLiNER-Bird"
    NuNER_Zero = "NuNER-Zero"
    NuNER_Zero_4K = "NuNER-Zero-4K"
    NuNER_Zero_span = "NuNER-Zero-span"

#-----

# schemas/auth.py
#-----
# schemas/auth.py

from pydantic import BaseModel

class User(BaseModel):
    username: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: str | None = None

#-----

# schemas/inference.py
#-----
# src/schemas/inference.py
from pydantic import BaseModel
from fastapi import Form
from typing import List, Dict
from datetime import datetime

class InferenceRequest(BaseModel):
    labels: List[str]
    threshold: float
    flat_ner: bool
    multi_label: bool
    batch_size: int

    @classmethod
    def as_form(
        cls,
        labels: str = Form("PERSON,PLACE,THING,ORGANIZATION,DATE,TIME", description="Types d'entités à extraire"),
        threshold: float = Form(0.3, description="Seuil de confiance pour l'inférence"),
        flat_ner: bool = Form(True, description="If need to extract parts of complex entities: False"),
        multi_label: bool = Form(False, description="If entities belong to several classes: True"),
        batch_size: int = Form(12, description="Taille du lot d'inférence")
    ) -> "InferenceRequest":
        # Les labels sont séparés par des virgules dans le formulaire, donc nous les convertissons en liste
        return cls(
            labels=labels.split(","),
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size
        )

class InferenceResponse(BaseModel):
    id: int
    file_path: str
    entities: List[Dict]
    created_at: datetime

    class Config:   
        from_attributes = True

#-----

# schemas/train.py
#-----
# src/schemas/train.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
class TrainRequest(BaseModel):
    dataset_id: int = Field(..., example=1)
    epochs: int = Field(10, example=20)
    batch_size: int = Field(32, example=64)

class TrainResponse(BaseModel):
    id: int
    run_id: str
    dataset_id: int
    epochs: int
    batch_size: int
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

#-----

# utils/file_utils.py
#-----
# src/utils/file_utils.py
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal
from pdf2image import convert_from_path
from pathlib import Path
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    @staticmethod
    def extract_text_and_bounding_boxes(file_path: Path) -> Optional[dict]:
        """Extract text and bounding boxes from a PDF using PDFMiner."""
        try:
            extracted_data = {"text": "", "boxes": []}
            for page_layout in extract_pages(str(file_path)):
                for element in page_layout:
                    if isinstance(element, LTTextBoxHorizontal):
                        for text_line in element:
                            if isinstance(text_line, LTTextLineHorizontal):
                                bbox = text_line.bbox  # Get the bounding box of the text
                                extracted_data["text"] += text_line.get_text()
                                extracted_data["boxes"].append({
                                    "text": text_line.get_text().strip(),
                                    "bbox": bbox,  # Coordinates (x0, y0, x1, y1)
                                    "page_num": page_layout.pageid
                                })
            return extracted_data
        except Exception as e:
            logger.error(f"Error extracting text and bounding boxes from PDF: {e}")
            return None

    @staticmethod
    def convert_pdf_to_images(file_path: Path, output_dir: Path) -> List[Path]:
        """Convert PDF pages to images using pdf2image."""
        try:
            images = convert_from_path(str(file_path))
            image_paths = []
            for i, image in enumerate(images):
                image_path = output_dir / f"{file_path.stem}_page_{i+1}.png"
                image.save(image_path, "PNG")
                image_paths.append(image_path)
            logger.info(f"Converted {file_path} to {len(images)} images.")
            return image_paths
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []

    @staticmethod
    def process_file(file_path: Path, output_dir: Path) -> dict:
        """Process a PDF file by extracting text and converting pages to images."""
        text_and_boxes = FileProcessor.extract_text_and_bounding_boxes(file_path)
        images = FileProcessor.convert_pdf_to_images(file_path, output_dir)
        return {"text": text_and_boxes["text"], "boxes": text_and_boxes["boxes"], "images": images}

#-----

# utils/mlflow_manager.py
#-----
# src/utils/mlflow_manager.py

import mlflow
import subprocess
from core.config import settings
from mlflow.exceptions import MlflowException
import logging

class MLflowManager:
    def __init__(self):
        self.tracking_uri = settings.mlflow_tracking_uri
        self.backend_store_uri = settings.mlflow_backend_store_uri
        self.artifact_root = settings.mlflow_artifact_root

    def setup_mlflow(self):
        # Définir l'URI de suivi MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            # Essayer de définir un experiment
            mlflow.set_experiment("GLiNER_Experiments")
            logging.info("MLflow experiment 'GLiNER_Experiments' is set.")
        except MlflowException as e:
            logging.warning(f"MLflow Exception: {e}")
            logging.info("Il semble que les tables MLflow soient manquantes. Tentative d'initialisation de la base de données.")
            self.run_migrations()
            # Réessayer de définir l'experiment après la création des tables
            mlflow.set_experiment("GLiNER_Experiments")
            logging.info("MLflow experiment 'GLiNER_Experiments' is set after migrations.")

    def run_migrations(self):
        try:
            # Exécuter la commande de migration MLflow
            logging.info("Exécution des migrations MLflow...")
            subprocess.run(
                [
                    "mlflow", "db", "upgrade",
                    self.backend_store_uri
                ],
                check=True
            )
            logging.info("Tables de la base de données MLflow créées avec succès.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Erreur lors de la création des tables MLflow : {e}")

#-----

# utils/__init__.py
#-----

#-----

# utils/gliner_utils.py
#-----
# src/utils/gliner_utils.py
import subprocess
import os
import json
from models.ner_model import NERModel

# Initialize model
ner_model = NERModel()

async def run_gliner_inference(text: str, labels: str):
    """
    Run inference on the provided text and labels using the NER model.
    """
    entities = ner_model.batch_predict([text], labels.split(","))
    return entities[0]  # Assuming a single result for now

def create_gliner_dataset(data: list, format: str = "json-ner") -> str:
    dataset_path = f"datasets/dataset.{format}"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w') as f:
        if format == "json-ner":
            json.dump(data, f, indent=4)
        elif format == "conllu":
            # Implémenter la conversion en CONLLU
            conllu_data = convert_to_conllu(data)
            f.write(conllu_data)
        else:
            raise ValueError("Format de dataset non supporté.")
    return dataset_path

def convert_to_conllu(data: list) -> str:
    # Implémenter la logique de conversion en CONLLU
    # Ceci est un exemple simplifié
    conllu_str = ""
    for item in data:
        token_id = item.get("token_id", 0)
        token = item.get("token", "")
        entity = item.get("entity", "O")
        conllu_str += f"{token_id}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{entity}\n"
    return conllu_str

def train_gliner_model(dataset_path: str, epochs: int, batch_size: int) -> str:
    # Exemple de commande pour entraîner GLiNER
    command = ["python", "gliner.py", "--train", dataset_path, "--epochs", str(epochs), "--batch_size", str(batch_size)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"GLiNER training failed: {result.stderr}")
    # Supposons que le run_id est retourné
    return result.stdout.strip()

#-----

# utils/metrics.py
#-----
# src/utils/metrics.py

from prometheus_client import Counter, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from fastapi import Response
# Utilisation d'un registre spécifique
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    'request_count', 'Total number of requests', ['method', 'endpoint', 'http_status'], registry=registry
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Latency of HTTP requests', 
    ['method', 'endpoint', 'http_status'],
    registry=registry
)

def setup_metrics(app):

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        REQUEST_LATENCY.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status=response.status_code
        ).inc()

        return response

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry=registry), media_type=CONTENT_TYPE_LATEST)

#-----

# utils/mlflow_setup.py
#-----
import mlflow
from core.config import settings

def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("GLiNER_Experiments")

#-----

# models/ner_model.py
#-----
# src/models/ner_model.py

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Dict, Union

import GPUtil
import torch
from torch.utils.data import DataLoader, Dataset
from gliner import GLiNER
from loguru import logger
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, get_cosine_schedule_with_warmup

from core.config import settings, MODELS, TRAIN_CONFIG, DEVICE


try:
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass

settings.train_config["device"] = DEVICE  # Update device in config

class NERDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class NERModel:
    """Named Entity Recognition model."""

    def __init__(
        self,
        name: str = "GLiNER-S",
        local_model_path: Optional[str] = None,
        overwrite: bool = False,
        train_config: dict = settings.train_config,
    ) -> None:
        """Initialize the NERModel."""
        if name not in MODELS:
            raise ValueError(f"Invalid model name: {name}")
        self.model_id: str = MODELS[name]

        # Create a models directory
        workdir = Path.cwd() / "models"
        workdir.mkdir(parents=True, exist_ok=True)
        if local_model_path is None:
            local_model_path = name
        else:
            local_model_path = (workdir / local_model_path).resolve()
        if Path(local_model_path).exists() and not overwrite:
            raise ValueError(f"Model path already exists: {str(local_model_path)}")

        self.local_model_path: Path = Path(local_model_path)

        # Set device
        self.device: str = train_config.get("device", DEVICE)
        logger.info(f"Device: [{self.device}]")

        # Define hyperparameters
        self.train_config: SimpleNamespace = SimpleNamespace(**train_config)

        # Initialize model as None for lazy loading
        self.model: Optional[GLiNER] = None

    def __load_model_remote(self) -> None:
        """Actually load the model."""
        self.model = GLiNER.from_pretrained(self.model_id)

    def __load_model_local(self) -> None:
        """Load the model from a local path."""
        try:
            local_model_path = str(self.local_model_path.resolve())
            self.model = GLiNER.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            logger.exception("Failed to load model from local path.", e)
            raise

    def load(self, mode: Literal["local", "remote", "auto"] = "auto") -> None:
        """Load the model."""
        if self.model is None:
            if mode == "local":
                self.__load_model_local()
            elif mode == "remote":
                self.__load_model_remote()
            elif mode == "auto":
                if self.local_model_path.exists():
                    self.__load_model_local()
                else:
                    self.__load_model_remote()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            GPUtil.showUtilization()
            logger.info(
                f"Loaded model: [{self.model_id}] | N Params: [{self.model_param_count}] | [{self.model_size_in_mb}]"
            )
        else:
            logger.warning("Model already loaded.")

        logger.info(f"Moving model weights to: [{self.device}]")
        self.model = self.model.to(self.device)

    @property
    def model_size_in_bytes(self) -> int:
        """Returns the approximate size of the model parameters in bytes."""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    @property
    def model_param_count(self) -> str:
        """Returns the number of model parameters in billions."""
        return f"{sum(p.numel() for p in self.model.parameters()) / 1e9:,.2f} B"

    @property
    def model_size_in_mb(self) -> str:
        """Returns the string repr of the model parameter size in MB."""
        return f"{self.model_size_in_bytes / 1024**2:,.2f} MB"

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Train the GLiNER model."""
        if self.model is None:
            self.load()

        GPUtil.showUtilization()

        # Prepare datasets
        train_dataset = NERDataset(train_data)
        eval_dataset = NERDataset(eval_data["samples"]) if eval_data else None

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.train_config.save_directory,
            num_train_epochs=self.train_config.epochs,
            per_device_train_batch_size=self.train_config.batch_size,
            per_device_eval_batch_size=self.train_config.batch_size,
            learning_rate=self.train_config.lr_others,
            evaluation_strategy="steps",
            eval_steps=self.train_config.eval_every,
            logging_dir=f"{self.train_config.save_directory}/logs",
            logging_steps=10,
            save_steps=self.train_config.eval_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            # Add other training arguments as needed
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=...,  # Define if needed
            # compute_metrics=...,  # Define if needed
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

    def batch_predict(
        self,
        targets: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[str]]:
        """Batch predict."""
        if self.model is None:
            self.load()

        self.model.eval()
        predictions = []
        for i, batch in enumerate(tqdm(self.chunk_list(targets, batch_size), desc="Predicting")):
            if i % 100 == 0:
                logger.debug(f"Predicting Batch [{i:,}]...")
            entities = self.model.batch_predict_entities(
                texts=batch,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            predictions.extend(entities)
        return predictions

    def save(self, file_name: str) -> None:
        """Save the model to a file."""
        self.model.save_pretrained(file_name)

    def test(self) -> None:
        """Test the model."""
        examples = ["hello John, your reservation is at 6pm"]
        predictions = self.model.batch_predict_entities(
            examples,
            labels=["Person", "Time"],
            threshold=0.5,
        )
        logger.info(predictions)

    @staticmethod
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Utility function to split a list into chunks."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

#-----

# models/__init__.py
#-----

#-----

# routers/ml_backend.py
#-----
# src/routers/ml_backend.py

from fastapi import APIRouter, Request, HTTPException
from typing import List, Dict
from services.labelstudio_manager import NERLabelStudioMLBackend

router = APIRouter()

# Instanciation du backend ML
ml_backend = NERLabelStudioMLBackend()

router = APIRouter(
    prefix="/ml_backend",
    tags=["ML Backend"]
)

@router.get("/health")
async def health():
    """Endpoint de santé pour vérifier si le backend ML est opérationnel."""
    try:
        # Vous pouvez ajouter des vérifications supplémentaires si nécessaire
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(request: Request):
    """Endpoint pour générer des prédictions."""
    try:
        tasks = await request.json()
        results = ml_backend.predict(tasks)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fit")
async def fit(request: Request):
    """Endpoint pour entraîner le modèle avec des annotations."""
    try:
        completions = await request.json()
        ml_backend.fit(completions)
        return {"status": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----

# routers/__init__.py
#-----

#-----

# routers/dataset.py
#-----
# src/routers/dataset.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from schemas.dataset import DatasetUpdate
from typing import List
from fastapi.responses import JSONResponse
from services.dataset_creator import create_ner_dataset
from schemas.dataset import DatasetResponse
from db.session import SessionLocal
from db.models import Dataset
from sqlalchemy.orm import Session
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
import time

router = APIRouter(
    prefix="/datasets",
    tags=["Dataset Creator"]
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    files: List[UploadFile] = File(...),
    name: str = Form(None),
    labels: str = Form("PERSON,ORG,GPE,DATE"),
    output_format: str = Form("json"),
    db: Session = Depends(get_db)
):
    """
    Endpoint to create an NER dataset from uploaded files, with optional name and format.
    If no name is provided, the name will default to the first PDF's filename + dataset ID.
    """
    start_time = time.time()

    try:
        # Convert labels string to list
        labels_list = [label.strip() for label in labels.split(',')] if labels else None

        # Create the dataset
        dataset_data = await create_ner_dataset(files, output_format=output_format, labels=labels_list, name=name, db=db)

        # Determine default name if not provided
        if not name:
            pdf_files = [file for file in files if file.filename.endswith('.pdf')]
            if pdf_files:
                default_name = pdf_files[0].filename.rsplit('.', 1)[0]  # Filename without extension
            else:
                default_name = "dataset"

            # Get next auto-increment ID from the database for naming
            next_id = db.execute("SELECT nextval('datasets_id_seq')").scalar()
            name = f"{default_name}_{next_id}"

        # Save dataset to the database
        dataset = Dataset(name=name, data=dataset_data)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        # Increment request count with the correct labels
        REQUEST_COUNT.labels(method="POST", endpoint="/datasets/", http_status="200").inc()

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )

    except Exception as e:
        db.rollback()
        REQUEST_COUNT.labels(method="POST", endpoint="/datasets/", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method="POST", endpoint="/datasets/", http_status="200").observe(latency)
        print(f"Dataset creation completed in {latency:.2f} seconds")

@router.get("/{dataset_id}", response_model=DatasetResponse)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_read").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="dataset_read").observe(latency)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        data=dataset.data,
        created_at=dataset.created_at.isoformat()
    )

@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(dataset_id: int, request: DatasetUpdate, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_update").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        dataset.name = request.name
        dataset.data = request.data
        db.commit()
        db.refresh(dataset)
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            data=dataset.data,
            created_at=dataset.created_at.isoformat()
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="dataset_update").observe(latency)

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="dataset_delete").inc()
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        db.delete(dataset)
        db.commit()
        return {"detail": "Dataset deleted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="dataset_delete").observe(latency)

#-----

# routers/auth.py
#-----
# src/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from services.security import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, fake_users_db
from schemas.auth import Token

router = APIRouter(tags=["Authentication"])

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Informations d'identification incorrectes",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

#-----

# routers/inference.py
#-----
# src/routers/inference.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, Form
from sqlalchemy.orm import Session
from pathlib import Path
from schemas.inference import InferenceRequest, InferenceResponse
from schemas.models_dict import ModelName
from db.session import SessionLocal
from db.models import Inference
from utils.file_utils import FileProcessor
from models.ner_model import NERModel, MODELS
from loguru import logger
import time
from typing import List


# Create an instance of the NERModel
ner_model_instance = NERModel(name="GLiNER-S")

router = APIRouter(
    prefix="/predict",
    tags=["Inference"]
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def process_file_and_inference(file_content: bytes, file_name: str, labels: str, threshold: float, db: Session, predictions: List):
    file_path = Path(f"/tmp/{file_name}")
    logger.info(f"Processing file: {file_name}")

    # Save file temporarily
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # Extract text from the file
    text = FileProcessor.process_file(file_path)
    if text is None:
        logger.error(f"Failed to extract text from {file_name}")
        raise HTTPException(status_code=400, detail="Failed to extract text from file")

    # Predict entities
    raw_entities = ner_model_instance.batch_predict([text], labels.split(","))
    logger.info(f"Predicted entities: {raw_entities}")

    # Flatten entities and format them properly
    entities = []
    for entity_group in raw_entities:
        for entity in entity_group:
            entities.append({
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": entity["label"],
                "score": entity["score"]
            })

    # Save inference in the database
    inference = Inference(file_path=file_name, entities=entities)
    db.add(inference)
    db.commit()
    db.refresh(inference)
    logger.info(f"Inference saved to DB with ID: {inference.id}")

    # Add to predictions list
    predictions.append(InferenceResponse(
        id=inference.id,
        file_path=inference.file_path,
        entities=inference.entities,
        created_at=inference.created_at.isoformat()
    ))

@router.post("/", response_model=List[InferenceResponse], status_code=202)
async def predict_endpoint(
    inference_request: InferenceRequest = Depends(InferenceRequest.as_form),
    selected_model: ModelName = Form(..., description="Sélectionnez le modèle NER"),
    files: List[UploadFile] = File(..., description="Fichiers à traiter"),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    predictions = []

    if selected_model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle {selected_model} non valide. Choisissez parmi: {', '.join(MODELS.keys())}")

    try:
        # Charger le modèle sélectionné
        logger.info(f"Chargement du modèle {selected_model}")
        ner_model_instance = NERModel(name=selected_model)
        ner_model_instance.load()  # Charger le modèle sélectionné

        # Process each file sequentially and synchronously
        for file in files:
            logger.info(f"Received file: {file.filename}")
            file_content = await file.read()  # Read file content
            process_file_and_inference(
                file_content, 
                file.filename, 
                ",".join(inference_request.labels), 
                inference_request.threshold, 
                db, 
                predictions
            )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        logger.info(f"Inference completed in {latency:.2f} seconds")

    logger.info(f"Returning predictions: {predictions}")
    return predictions

#-----

# routers/train.py
#-----
# src/routers/train.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from schemas.train import TrainRequest, TrainResponse
from models.ner_model import NERModel
from db.session import SessionLocal
from db.models import TrainingRun, Dataset
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from sqlalchemy.orm import Session
import time
import uuid
from loguru import logger

router = APIRouter(
    prefix="/train",
    tags=["Train"]
)

# Dependency pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=TrainResponse)
def train_endpoint(request: TrainRequest, db: Session = Depends(get_db)):
    """Endpoint pour entraîner le modèle NER."""
    start_time = time.time()
    run_id = str(uuid.uuid4())  # Génère un ID de run unique

    try:
        # Récupérer le dataset depuis la DB
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset non trouvé")

        # Vérifier que le dataset contient des données d'entraînement et d'évaluation
        train_data = dataset.train_data  # Assurez-vous que 'train_data' est une liste de dicts
        eval_data = dataset.eval_data  # Assurez-vous que 'eval_data' est un dict avec 'samples'

        if not train_data:
            raise HTTPException(status_code=400, detail="Données d'entraînement vides")
        if eval_data is None or 'samples' not in eval_data:
            raise HTTPException(status_code=400, detail="Données d'évaluation manquantes ou incomplètes")

        # Préparer la configuration d'entraînement
        train_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "save_directory": "checkpoints",  # Exemple, ajustez selon vos besoins
            "lr_others": 5e-5,  # Exemple, ajustez selon vos besoins
            "warmup_ratio": 0.1,
            "max_types": 25,
            "shuffle_types": True,
            "random_drop": True,
            "max_neg_type_ratio": 1,
            "max_len": 384,
            "eval_every": 100
        }

        # Initialiser le NERModel avec les paramètres d'entraînement
        ner_model = NERModel(
            name=dataset.model_name,  # Assurez-vous que le dataset a un champ 'model_name'
            train_config=train_config
        )

        # Entraîner le modèle
        ner_model.train(train_data=train_data, eval_data=eval_data)

        # Enregistrer le run d'entraînement dans la DB
        training_run = TrainingRun(
            run_id=run_id,  # Utiliser l'ID de run généré
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status="Completed"
        )
        db.add(training_run)
        db.commit()
        db.refresh(training_run)

        # Incrémenter les métriques Prometheus pour une réponse réussie
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status="200").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status="200").observe(time.time() - start_time)

        return TrainResponse(
            id=training_run.id,
            run_id=training_run.run_id,
            dataset_id=training_run.dataset_id,
            epochs=training_run.epochs,
            batch_size=training_run.batch_size,
            status=training_run.status,
            created_at=training_run.created_at.isoformat()
        )

    except HTTPException as http_exc:
        # Enregistrer un run d'entraînement échoué
        training_run = TrainingRun(
            run_id=run_id,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status=f"Failed: {http_exc.detail}"
        )
        db.add(training_run)
        db.commit()

        # Incrémenter les métriques Prometheus pour une réponse échouée
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status=str(http_exc.status_code)).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status=str(http_exc.status_code)).observe(time.time() - start_time)

        raise http_exc

    except Exception as e:
        # Enregistrer un run d'entraînement échoué avec une exception non HTTP
        training_run = TrainingRun(
            run_id=run_id,
            dataset_id=request.dataset_id,
            epochs=request.epochs,
            batch_size=request.batch_size,
            status=f"Failed: {str(e)}"
        )
        db.add(training_run)
        db.commit()

        # Incrémenter les métriques Prometheus pour une réponse échouée
        REQUEST_COUNT.labels(method="POST", endpoint="train", http_status="500").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="train", http_status="500").observe(time.time() - start_time)

        logger.error(f"Erreur lors de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=str(e))

#-----

# services/ocr_service.py
#-----
# services/ocr_service.py

import easyocr
from pathlib import Path

reader = easyocr.Reader(['en', 'fr'])  # Vous pouvez spécifier les langues nécessaires

def extract_text_from_image(file_path: Path) -> str:
    result = reader.readtext(str(file_path), detail=0, paragraph=True)
    text = ' '.join(result)
    return text

#-----

# services/dataset_creator.py
#-----
# src/services/dataset_creator.py
from typing import List, Dict, Optional
from fastapi import UploadFile
from pathlib import Path
from utils.file_utils import FileProcessor
from minio import Minio
from sqlalchemy.orm import Session
from db.models import Dataset
from core.config import settings
import os

# Setup MinIO client
minio_client = Minio(
    "localhost:9000",  # settings.minio_api_url,
    access_key=settings.minio_root_user,
    secret_key=settings.minio_root_password,
    secure=False  # Set to True if using HTTPS
)

bucket_name = "datasets"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

async def upload_to_minio(file_path: Path, name: str, file_type: str) -> str:
    """Upload a file to MinIO and return its URL."""
    object_name = f"{name}/{file_type}/{file_path.name}"
    minio_client.fput_object(bucket_name, object_name, str(file_path))
    file_url = f"http://{settings.minio_api_url}/{bucket_name}/{object_name}"
    return file_url

async def create_ner_dataset(
    files: List[UploadFile],
    output_format: str,
    labels: Optional[List[str]],
    name: str,
    db: Session
) -> List[Dict]:
    """Create a dataset for NER tasks from the provided PDF files and store in MinIO."""
    dataset = []
    output_dir = Path(f"/tmp/extracted_data/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = output_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the file: extract text and bounding boxes, convert PDF to images
        file_data = FileProcessor.process_file(file_path, output_dir)

        # Upload images to MinIO and get their URLs
        image_urls = []
        for image_path in file_data["images"]:
            image_url = await upload_to_minio(image_path, name, "images")
            image_urls.append(image_url)

        # Upload text file to MinIO (optional, if you want the text separately)
        text_file_path = output_dir / f"{file.filename}.txt"
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(file_data["text"])
        text_url = await upload_to_minio(text_file_path, name, "text")

        # Prepare the dataset entry for Label Studio
        dataset.append({
            'task_id': len(dataset) + 1,  # Unique task ID
            'image_urls': image_urls,
            'text_url': text_url,
            'boxes': file_data["boxes"],  # Bounding boxes of text
            'file_name': file.filename,
            'annotations': []  # No annotations initially
        })

    # Save the dataset metadata in PostgreSQL
    dataset_entry = Dataset(name=name, data=dataset)
    db.add(dataset_entry)
    db.commit()
    db.refresh(dataset_entry)

    # Generate tasks in Label Studio format
    label_studio_tasks = format_for_label_studio(dataset)
    return label_studio_tasks

def format_for_label_studio(dataset: List[Dict]) -> List[Dict]:
    """Convert the dataset into a format compatible with Label Studio."""
    label_studio_dataset = []
    for data in dataset:
        # Create a task for each image in the PDF
        for image_url, box in zip(data["image_urls"], data["boxes"]):
            task = {
                "data": {
                    "image": image_url,  # Image URL for Label Studio to display
                    "text": data["text_url"]  # Text associated with the image (in MinIO)
                },
                "meta": {
                    "source": data["file_name"],
                    "task_id": data["task_id"]
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "from_name": "bbox",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "value": {
                                    "x": box['bbox'][0],
                                    "y": box['bbox'][1],
                                    "width": box['bbox'][2] - box['bbox'][0],
                                    "height": box['bbox'][3] - box['bbox'][1]
                                }
                            }
                        ]
                    }
                ]
            }
            label_studio_dataset.append(task)
    return label_studio_dataset

#-----

# services/model_manager.py
#-----
# src/services/model_manager.py
from models.ner_model import NERModel
from typing import List
from threading import Lock
from typing import Optional
from loguru import logger

class ModelManager:
    """Gestionnaire du modèle NER."""

    def __init__(self):
        self.model: Optional[NERModel] = None
        self.lock = Lock()

    def load_model(self, name: str = "GLiNER-S") -> None:
        with self.lock:
            if self.model is None:
                logger.info("Chargement du modèle NER...")
                self.model = NERModel(name=name)
                self.model.load()
                logger.info("Modèle NER chargé.")
            else:
                logger.info("Le modèle est déjà chargé.")

    def predict(
        self,
        texts: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[dict]]:
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        return self.model.batch_predict(
            targets=texts,
            labels=labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
        )

    def train_model(
        self,
        train_data: List[dict],
        eval_data: dict = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        self.model.train(train_data, eval_data)

#-----

# services/labelstudio_manager.py
#-----
# src/services/labelstudio_manager.py

from typing import List, Dict
from services.model_manager import ModelManager
import logging

class NERLabelStudioMLBackend:
    def __init__(self):
        self.model_manager = ModelManager()

    def start(self):
        """Initialise et charge le modèle."""
        logging.info("Initialisation du backend ML...")
        self.model_manager.load_model()
        logging.info("Backend ML prêt.")

    def stop(self):
        """Nettoie les ressources si nécessaire."""
        logging.info("Arrêt du backend ML...")
        # Implémentez la logique de nettoyage si nécessaire
        logging.info("Backend ML arrêté.")

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """Génère des prédictions pour les tâches fournies."""
        texts = [task['data']['text'] for task in tasks]
        predictions = self.model_manager.predict(
            texts=texts,
            labels=None,  # Spécifiez les étiquettes si nécessaire
            flat_ner=True,
            threshold=0.3,
            multi_label=False,
            batch_size=12
        )

        results = []
        for task, prediction in zip(tasks, predictions):
            result = []
            for entity in prediction:
                result.append({
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": entity['start'],
                        "end": entity['end'],
                        "labels": [entity['label']]
                    }
                })
            results.append({"result": result})

        return results

    def fit(self, completions: List[Dict], **kwargs):
        """Entraîne le modèle avec les annotations fournies."""
        annotated_data = []
        for completion in completions:
            text = completion['data']['text']
            annotations = completion['annotations']
            entities = []
            for ann in annotations:
                for label in ann['value']['labels']:
                    entities.append({
                        'start': ann['value']['start'],
                        'end': ann['value']['end'],
                        'label': label
                    })
            annotated_data.append({
                'text': text,
                'entities': entities
            })
        
        self.model_manager.train_model(train_data=annotated_data)

#-----

# services/security.py
#-----
# services/security.py

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

from schemas.auth import User, UserInDB, TokenData

# Configuration simplifiée pour l'exemple
SECRET_KEY = "your_secret_key" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Création du contexte pour le hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Utilisateurs fictifs pour l'exemple
fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": pwd_context.hash("wonderland"),
        "disabled": False,
    },
    "bob": {
        "username": "bob",
        "hashed_password": pwd_context.hash("builder"),
        "disabled": False,
    },
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Par défaut, le token expire dans 15 minutes
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les informations d'identification",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

#-----

# db/models.py
#-----
# src/db/models
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, event
from sqlalchemy.sql import func
from .base import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    data = Column(JSON, nullable=False)  # Stockage des données NER en JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
# Function to generate the name after the dataset is inserted

@event.listens_for(Dataset, 'after_insert')
def generate_dataset_name(mapper, connection, target):
    if target.name == "":  # or handle None if necessary
        connection.execute(
            Dataset.__table__.update().
            where(Dataset.id == target.id).
            values(name=f"dataset_{target.id}")
        )
class Inference(Base):
    __tablename__ = "inferences"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    entities = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, nullable=False)
    dataset_id = Column(Integer, nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=32)
    status = Column(String, default="Started")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

#-----

# db/__init__.py
#-----

#-----

# db/session.py
#-----
# src/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Fonction get_db pour récupérer une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
#-----

# db/init_db.py
#-----
# src/db/init_db.py
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import sessionmaker
from db.session import engine
from db.models import Base

def init_db():
    # Create tables if they don't exist (for legacy usage)
    Base.metadata.create_all(bind=engine)

    # Run migrations using Alembic
    alembic_cfg = Config("alembic.ini") 
    command.upgrade(alembic_cfg, "head")  # Run all migrations

#-----

# db/base.py
#-----
# src/db/base.py
from sqlalchemy.orm import declarative_base

Base = declarative_base()

#-----
