import os
import re
import json
import random
import shutil
import zipfile
import torch
from typing import List, Dict, Union, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.data_processing import GLiNERDataset

app = FastAPI(title="GLiNER API", description="API for GLiNER functionalities", version="1.0")

# Ensure directories exist
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("data"):
    os.makedirs("data")

# List of available models
AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Global variables
annotator = None
model_generator = None


# Pydantic models avec exemples pour Swagger
class TextInput(BaseModel):
    text: str = Field(..., example="IBM Watson defeated human champions in the game of Jeopardy!")

class NERInput(BaseModel):
    model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    labels: Optional[str] = Field(None, example="person, organization, location")
    threshold: Optional[float] = Field(0.5, example=0.5)
    nested_ner: Optional[bool] = Field(False, example=False)

class NEROutput(BaseModel):
    text: str
    entities: List[Dict[str, Union[str, int, float]]] = Field(..., example=[
        {"entity": "organization", "word": "IBM", "start": 0, "end": 3, "score": 0.98}
    ])

class AnnotateInput(BaseModel):
    model: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    labels: str = Field(..., example="person, organization, location")
    threshold: float = Field(0.5, example=0.5)
    prompt: Optional[str] = Field(None, example="Please annotate the following text:")
    sentences: List[str] = Field(..., example=["Google is building a new office in New York."])

class TrainInput(BaseModel):
    model_name: str = Field(..., example="knowledgator/gliner-multitask-large-v0.5")
    custom_model_name: str = Field(..., example="my-custom-model")
    train_data: str  # Path to the training data
    split_ratio: float = Field(0.9, example=0.9)
    learning_rate: float = Field(5e-6, example=5e-6)
    weight_decay: float = Field(0.01, example=0.01)
    batch_size: int = Field(8, example=8)
    epochs: int = Field(1, example=1)
    compile_model: bool = Field(False, example=False)

class EvaluateInput(BaseModel):
    model_name: str = Field(..., example="my-custom-model")

class EvaluateOutput(BaseModel):
    f1_score: float = Field(..., example=0.85)
    results: str = Field(..., example="Entity-wise F1 score: ...")

# Helper functions
def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['entity'] == current['entity'] and (
            next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']
        ):
            current['word'] += ' ' + next_entity['word']
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged

def annotate_text(
    model, text, labels: List[str], threshold: float, nested_ner: bool
) -> Dict:
    labels = [label.strip() for label in labels]
    r = {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }
    r["entities"] = merge_entities(r["entities"])
    return r

class AutoAnnotator:
    def __init__(
        self, model_name: str = "knowledgator/gliner-multitask-large-v0.5",
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ) -> None:
        self.model = GLiNER.from_pretrained(model_name).to(device)
        self.annotated_data = []
        self.stat = {
            "total": None,
            "current": -1
        }

    def auto_annotate(
        self, data: List[str], labels: List[str],
        prompt: Optional[str] = None, threshold: float = 0.5, nested_ner: bool = False
    ) -> List[Dict]:
        self.stat["total"] = len(data)
        self.stat["current"] = -1  # Reset current progress
        for text in data:
            self.stat["current"] += 1
            if isinstance(prompt, list):
                prompt_text = random.choice(prompt)
            else:
                prompt_text = prompt
            text_with_prompt = f"{prompt_text}\n{text}" if prompt_text else text

            annotation = annotate_text(self.model, text_with_prompt, labels, threshold, nested_ner)

            if not annotation["entities"]:  # If no entities identified
                annotation = {"text": text, "entities": []}

            self.annotated_data.append(annotation)
        return self.annotated_data

class ModelGenerator:
    def __init__(self) -> None:
        self.previous_path = None
        self.path = None
        self.model = None

    def get_model(self, path):
        if self.path != path:
            self.model = GLiNER.from_pretrained(path, load_tokenizer=True).to(device)
            self.path = path
        return self.model

model_generator = ModelGenerator()

# API Endpoints
@app.post("/ner/", response_model=NEROutput)
def ner_endpoint(input_data: NERInput):
    model_path = f"models/{input_data.model_name}"
    if not os.path.exists(model_path):
        if input_data.model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(input_data.model_name).to(device)
        else:
            raise HTTPException(status_code=404, detail="Model not found.")
    else:
        model = GLiNER.from_pretrained(model_path).to(device)

    labels = [label.strip() for label in input_data.labels.split(",")] if input_data.labels else None
    result = annotate_text(
        model, input_data.text, labels, input_data.threshold, input_data.nested_ner
    )
    return result

@app.post("/annotate/")
def annotate_endpoint(input_data: AnnotateInput):
    try:
        labels = [label.strip() for label in input_data.labels.split(",")]
        annotator = AutoAnnotator(input_data.model)
        annotated_data = annotator.auto_annotate(
            input_data.sentences, labels, input_data.prompt, input_data.threshold
        )
        with open("data/annotated_data.json", "wt") as file:
            json.dump(annotated_data, file)
        return {"message": "Successfully annotated and saved as data/annotated_data.json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_dataset/")
def upload_dataset(file: UploadFile = File(...)):
    save_path = os.path.join("data", file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"File saved to {save_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
def train_endpoint(train_input: TrainInput):
    def load_and_prepare_data(train_path, split_ratio):
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"The file {train_path} does not exist.")

        with open(train_path, "r") as f:
            data = json.load(f)
        random.seed(42)
        random.shuffle(data)
        train_data = data[:int(len(data) * split_ratio)]
        test_data = data[int(len(data) * split_ratio):]
        return train_data, test_data

    def create_models_directory():
        if not os.path.exists("models"):
            os.makedirs("models")

    try:
        create_models_directory()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if train_input.model_name in AVAILABLE_MODELS:
            model = GLiNER.from_pretrained(train_input.model_name)
        else:
            model_path = f"models/{train_input.model_name}"
            if os.path.exists(model_path):
                model = GLiNER.from_pretrained(model_path)
            else:
                raise HTTPException(status_code=404, detail="Model not found.")

        train_data, test_data = load_and_prepare_data(train_input.train_data, train_input.split_ratio)

        with open("data/test.json", "wt") as file:
            json.dump(test_data, file)

        train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
        test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
        data_collator = DataCollatorWithPadding(model.config)

        if train_input.compile_model:
            torch.set_float32_matmul_precision('high')
            model.to(device)
            model.compile_for_training()
        else:
            model.to(device)

        training_args = TrainingArguments(
            output_dir="models",
            learning_rate=train_input.learning_rate,
            weight_decay=train_input.weight_decay,
            others_lr=train_input.learning_rate,
            others_weight_decay=train_input.weight_decay,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            per_device_train_batch_size=train_input.batch_size,
            per_device_eval_batch_size=train_input.batch_size,
            num_train_epochs=train_input.epochs,
            evaluation_strategy="epoch",
            save_steps=1000,
            save_total_limit=10,
            dataloader_num_workers=8,
            use_cpu=(device == torch.device('cpu')),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        model.save_pretrained(f"models/{train_input.custom_model_name}")

        return {"message": "Training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/", response_model=EvaluateOutput)
def evaluate_endpoint(evaluate_input: EvaluateInput):
    try:
        model_path = f"models/{evaluate_input.model_name}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found.")

        model = GLiNER.from_pretrained(model_path, load_tokenizer=True, local_files_only=True)

        with open('data/test.json', 'r') as file:
            test_data = json.load(file)

        with open('data/annotated_data.json', 'r') as file:
            annotated_data = json.load(file)

        # Extract all labels from each example
        all_labels = []
        for example in annotated_data:
            ner_data = example.get("ner", [])
            for entity in ner_data:
                label = entity[2]  # Assuming the label is the third element in the entity list
                if label not in all_labels:
                    all_labels.append(label)

        def get_for_one_path(test_dataset, entity_types):
            # evaluate the model
            results, f1 = model.evaluate(
                test_dataset, flat_ner=True, threshold=0.5, batch_size=12, entity_types=entity_types
            )
            return results, f1

        results, f1 = get_for_one_path(test_data, all_labels)
        return EvaluateOutput(f1_score=f1, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/zip_and_upload/")
def zip_and_upload_endpoint(model_name: str, drive_path: str):
    def zip_directory(model_name):
        model_path = f"models/{model_name}"
        zip_path = f"{model_path}.zip"

        if os.path.exists(model_path):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=model_path)
                        zipf.write(file_path, arcname)
            return zip_path
        else:
            return None

    def upload_to_drive(zip_path, drive_folder='My Drive'):
        # Placeholder for actual upload logic
        if zip_path and os.path.exists(zip_path):
            destination_dir = os.path.join(drive_folder)
            os.makedirs(destination_dir, exist_ok=True)
            destination = os.path.join(destination_dir, os.path.basename(zip_path))
            shutil.move(zip_path, destination)
            return f"File uploaded to {destination}"
        else:
            return "Zip file not found."

    try:
        zip_path = zip_directory(model_name)
        if zip_path:
            upload_message = upload_to_drive(zip_path, drive_folder=drive_path)
            return {"message": f"Directory '{model_name}' zipped successfully as '{zip_path}'. {upload_message}"}
        else:
            raise HTTPException(status_code=404, detail=f"Directory '{model_name}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)