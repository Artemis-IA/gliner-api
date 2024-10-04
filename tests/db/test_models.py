# tests/db/test_models.py
from src.db.models import Dataset, Inference, TrainingRun

def test_dataset_model():
    dataset = Dataset(name="Test Dataset", data=[{"text": "Sample", "entities": ["Entity1"]}])
    assert dataset.name == "Test Dataset"
    assert dataset.data == [{"text": "Sample", "entities": ["Entity1"]}]

def test_inference_model():
    inference = Inference(file_path="path/to/file.pdf", entities=[{"entity": "Entity1"}])
    assert inference.file_path == "path/to/file.pdf"
    assert inference.entities == [{"entity": "Entity1"}]

def test_training_run_model():
    training_run = TrainingRun(run_id="12345", dataset_id=1, epochs=10, batch_size=32)
    assert training_run.run_id == "12345"
    assert training_run.dataset_id == 1
    assert training_run.epochs == 10
    assert training_run.batch_size == 32
    assert training_run.status == "Started"
