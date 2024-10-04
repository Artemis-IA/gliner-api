# tests/services/test_ocr_service.py
from src.services.ocr_service import extract_text_from_image, extract_text_from_pdf

def test_extract_text_from_image():
    text = extract_text_from_image("tests/sample.png")
    assert isinstance(text, str)
    assert len(text) > 0
