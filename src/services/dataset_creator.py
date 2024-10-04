# services/dataset_creator.py

from typing import List, Dict
from pathlib import Path
from utils.file_utils import extract_text_from_pdf, extract_text_from_image
from src.schemas.dataset import Entity 
import spacy
import random
import json
import os

# Charger le modèle de langue pour l'annotation automatique
nlp = spacy.load('fr_core_news_sm')  # Assurez-vous d'installer le modèle spaCy pour le français

def create_ner_dataset(files: List[Path], output_format: str = 'json') -> List[Dict]:
    """
    Crée un dataset NER à partir des fichiers fournis.

    Args:
        files (List[Path]): Liste des chemins des fichiers.
        output_format (str): Format de sortie ('json', 'nner', 'conllu').

    Returns:
        List[Dict]: Liste des exemples avec texte et annotations.
    """
    dataset = []
    for file_path in files:
        if file_path.suffix == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix in ['.png', '.jpg', '.jpeg']:
            text = extract_text_from_image(file_path)
        else:
            continue  # Ignorer les types de fichiers non supportés

        # Annoter le texte en utilisant spaCy pour obtenir des annotations automatiques
        annotations = annotate_text(text)

        example = {
            'text': text,
            'annotations': annotations
        }

        dataset.append(example)

    # Sauvegarder le dataset dans le format souhaité
    if output_format == 'json':
        save_as_json(dataset)
    elif output_format == 'nner':
        save_as_nner(dataset)
    elif output_format == 'conllu':
        save_as_conllu(dataset)
    else:
        raise ValueError("Format de sortie non supporté. Choisissez parmi 'json', 'nner', 'conllu'.")

    return dataset

def annotate_text(text: str) -> List[Dict]:
    """
    Annoter le texte en utilisant spaCy pour obtenir des entités nommées.

    Args:
        text (str): Texte à annoter.

    Returns:
        List[Dict]: Liste des annotations avec 'start', 'end', et 'label'.
    """
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
    """
    Sauvegarder le dataset au format JSON.

    Args:
        dataset (List[Dict]): Dataset à sauvegarder.
        output_dir (str): Répertoire de sortie.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / 'dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dataset sauvegardé au format JSON dans {output_path}")

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
