# utils/huggingface_helper.py
from huggingface_hub import HfApi
from loguru import logger
from typing import Optional


class HuggingFaceHelper:
    def __init__(self):
        self.api = HfApi()

    def fetch_model_metadata(self, model_id: str) -> dict:
        """Fetch model metadata from Hugging Face."""
        try:
            model_info = self.api.model_info(model_id)
            return {
                "version": model_info.sha,
                "tags": model_info.tags,
                "description": self.fetch_readme(model_id) or "No description available.",
            }
        except Exception as e:
            logger.error(f"Error fetching metadata for model {model_id}: {e}")
            return {}

    def fetch_readme(self, model_id: str) -> Optional[str]:
        """Fetch README content for the model."""
        try:
            return self.api.model_info(model_id).cardData.get("model_card", "")
        except Exception as e:
            logger.warning(f"Unable to fetch README for model {model_id}: {e}")
            return None
