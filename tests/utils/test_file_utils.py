# tests/utils/test_file_utils.py
from src.utils.file_utils import extract_text_from_pdf, extract_text_from_image

def test_extract_text_from_pdf():
    text = extract_text_from_pdf("tests/sample.pdf")
    assert isinstance(text, str)
    assert "Some expected text" in text

def test_extract_text_from_image():
    text = extract_text_from_image("tests/sample.png")
    assert isinstance(text, str)
    assert "Some expected text" in text
