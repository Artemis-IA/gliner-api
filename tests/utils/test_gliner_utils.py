# tests/utils/test_gliner_utils.py
from src.utils.gliner_utils import run_gliner_inference, create_gliner_dataset

def test_run_gliner_inference():
    result = run_gliner_inference("tests/sample.pdf")
    assert isinstance(result, list)
    assert len(result) > 0

def test_create_gliner_dataset():
    data = [{"text": "Sample text", "annotations": [{"start": 0, "end": 6, "label": "Entity1"}]}]
    path = create_gliner_dataset(data, format="json-ner")
    assert path.endswith("dataset.json")
    # Optionnel : Vérifiez le contenu du fichier
