# tests/routers/test_train.py
def test_train_model(client, db):
    # Assurez-vous qu'un dataset avec ID 1 existe
    payload = {
        "dataset_id": 1,
        "epochs": 10,
        "batch_size": 32
    }
    response = client.post("/train/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "Completed"
    assert "created_at" in data
