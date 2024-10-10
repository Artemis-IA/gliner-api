# src/utils/file_utils.py
from pdfminer.high_level import extract_text
from PIL import Image
from services.ocr_service import extract_text_from_image
from pathlib import Path
from loguru import logger
from typing import Optional

class FileProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> Optional[str]:
        """Extract text from a PDF using PDFMiner."""
        try:
            text = extract_text(str(file_path))
            if not text.strip():
                logger.warning(f"Empty or unreadable PDF: {file_path}")
                return None
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None

    @staticmethod
    def extract_text_from_image(file_path: Path) -> Optional[str]:
        """Extract text from an image using Tesseract OCR."""
        try:
            image = Image.open(file_path)
            text = extract_text_from_image.image_to_string(image)
            if not text.strip():
                logger.warning(f"Empty or unreadable image: {file_path}")
                return None
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return None

    @staticmethod
    def process_file(file_path: Path) -> Optional[str]:
        """Determine file type and extract text accordingly."""
        if file_path.suffix.lower() == ".pdf":
            return FileProcessor.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff"]:
            return FileProcessor.extract_text_from_image(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
