# services/dataset_creator.py
from typing import List, Dict
from pathlib import Path
from utils.file_utils import FileProcessor
import spacy
import json
import os

nlp = spacy.load('fr_core_news_sm')  # Ensure spaCy is installed with 'fr_core_news_sm' model

def create_ner_dataset(files: List[Path], output_format: str = 'json') -> List[Dict]:
    """Create an NER dataset from the provided files."""
    dataset = []
    for file_path in files:
        if file_path.suffix == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
        elif file_path.suffix in ['.png', '.jpg', '.jpeg']:
            text = FileProcessor.extract_text_from_image(file_path)
        else:
            continue  # Ignore unsupported files

        # Annotate the text using spaCy to extract entities
        annotations = annotate_text(text)
        dataset.append({
            'text': text,
            'annotations': annotations
        })

    # Save the dataset in the desired format
    if output_format == 'json':
        save_as_json(dataset)
    elif output_format == 'nner':
        save_as_nner(dataset)
    elif output_format == 'conllu':
        save_as_conllu(dataset)
    else:
        raise ValueError("Unsupported output format. Choose 'json', 'nner', or 'conllu'.")

    return dataset

def annotate_text(text: str) -> List[Dict]:
    """Annotate text using spaCy NER to get named entities."""
    doc = nlp(text)
    annotations = []
    for ent in doc.ents:
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
