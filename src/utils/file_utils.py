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
