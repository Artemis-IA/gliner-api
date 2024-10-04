# tests/services/test_dataset_creator.py
from src.services.dataset_creator import create_ner_dataset

def test_create_ner_dataset():
    files = ["tests/sample.pdf", "tests/sample.png"]
    dataset = create_ner_dataset(files, output_format="json")
    assert len(dataset) == 2
    for example in dataset:
        assert "text" in example
        assert "annotations" in example
