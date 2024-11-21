# services/ocr_service.py

import easyocr
from pathlib import Path

reader = easyocr.Reader(['en', 'fr'])  # Vous pouvez spécifier les langues nécessaires

def extract_text_from_image(file_path: Path) -> str:
    result = reader.readtext(str(file_path), detail=0, paragraph=True)
    text = ' '.join(result)
    return text
