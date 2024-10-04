# tests/routers/test_inference.py
def test_create_inference(client):
    payload = {
        "file_path": "path/to/file.pdf"
    }
    response = client.post("/inference/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["file_path"] == "path/to/file.pdf"
    assert "entities" in data
    assert "id" in data
    assert "created_at" in data
