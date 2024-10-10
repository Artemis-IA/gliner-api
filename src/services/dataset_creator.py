# src/services/dataset_creator.py
from typing import List, Dict, Optional
from fastapi import UploadFile
from pathlib import Path
from utils.file_utils import FileProcessor
import spacy
import json
import os

# Load the spaCy NER model (French in this case)
nlp = spacy.load('fr_core_news_sm')

async def create_ner_dataset(
    files: List[UploadFile],
    output_format: str = 'json',
    labels: Optional[List[str]] = None
) -> List[Dict]:
    """Create an NER dataset from the provided files."""
    dataset = []
    
    for file in files:
        file_path = Path(f"/tmp/{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Extract text from the file
        text = FileProcessor.process_file(file_path)
        
        # Annotate text using spaCy NER
        annotations = annotate_text(text, labels)
        
        dataset.append({
            'text': text,
            'annotations': annotations
        })
    
    # Return the dataset in the desired format
    if output_format == 'json':
        return dataset  # Return as JSON structured data
    elif output_format == 'nner':
        return save_as_nner(dataset)
    elif output_format == 'conllu':
        return save_as_conllu(dataset)
    else:
        raise ValueError("Unsupported format. Choose 'json', 'nner', or 'conllu'.")


def annotate_text(text: str, labels: Optional[List[str]] = None) -> List[Dict]:
    """Annotate text using spaCy's NER model to get named entities."""
    doc = nlp(text)
    annotations = []
    for ent in doc.ents:
        if labels is None or ent.label_ in labels:
            annotations.append({
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
    return annotations



def save_as_json(dataset: List[Dict], output_dir: str = 'datasets') -> None:
    """Save the dataset as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / 'dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved as JSON at {output_path}")


def save_as_nner(dataset: List[Dict], output_dir: str = 'datasets') -> None:
    """
    Sauvegarder le dataset au format nNER.

    Args:
        dataset (List[Dict]): Dataset à sauvegarder.
        output_dir (str): Répertoire de sortie.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / 'dataset.nner'
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            text = example['text']
            annotations = example['annotations']
            entities = []
            for ann in annotations:
                entities.append(f"{ann['start']} {ann['end']} {ann['label']}")
            entities_str = '\n'.join(entities)
            f.write(f"{text}\n{entities_str}\n\n")
    print(f"Dataset sauvegardé au format nNER dans {output_path}")

def save_as_conllu(dataset: List[Dict], output_dir: str = 'datasets') -> None:
    """
    Sauvegarder le dataset au format CoNLL-U.

    Args:
        dataset (List[Dict]): Dataset à sauvegarder.
        output_dir (str): Répertoire de sortie.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / 'dataset.conllu'
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            text = example['text']
            annotations = example['annotations']
            tokens = nlp(text)
            for token in tokens:
                # Trouver l'étiquette correspondante
                label = 'O'
                for ann in annotations:
                    if token.idx >= ann['start'] and token.idx + len(token) <= ann['end']:
                        label = f"B-{ann['label']}"
                        break
                f.write(f"{token.text}\t{label}\n")
            f.write('\n')
    print(f"Dataset sauvegardé au format CoNLL-U dans {output_path}")
