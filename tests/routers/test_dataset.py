# tests/routers/test_dataset.py
def test_create_dataset(client):
    payload = {
        "name": "Test Dataset",
        "data": [{"text": "Example text", "entities": ["Entity1"]}]
    }
    response = client.post("/datasets/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Dataset"
    assert data["data"] == [{"text": "Example text", "entities": ["Entity1"]}]
    assert "id" in data
    assert "created_at" in data

def test_read_dataset(client):
    # Supposons que l'ID 1 existe
    response = client.get("/datasets/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert "name" in data
    assert "data" in data

def test_update_dataset(client):
    payload = {
        "name": "Updated Dataset",
        "data": [{"text": "Updated text", "entities": ["Entity2"]}]
    }
    response = client.put("/datasets/1", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Dataset"
    assert data["data"] == [{"text": "Updated text", "entities": ["Entity2"]}]

def test_delete_dataset(client):
    response = client.delete("/datasets/1")
    assert response.status_code == 200
    assert response.json()["detail"] == "Dataset deleted"
