# utils/file_utils.py

from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract
from pathlib import Path
from loguru import logger

def extract_text_from_pdf(file_path: Path) -> str:
    try:
        text = extract_text(str(file_path))
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte du PDF : {e}")
        text = ""
    return text

def extract_text_from_image(file_path: Path) -> str:
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte de l'image : {e}")
        text = ""
    return text
