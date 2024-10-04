# tests/routers/test_auth.py
from fastapi.testclient import TestClient
from src.routers.auth import router
from src.main import app

client = TestClient(app)

def test_login_success():
    response = client.post("/token", data={"username": "alice", "password": "wonderland"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_failure():
    response = client.post("/token", data={"username": "alice", "password": "wrongpassword"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Informations d'identification incorrectes"
