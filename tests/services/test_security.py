# tests/services/test_security.py
from src.services.security import authenticate_user, create_access_token, verify_password

def test_verify_password():
    assert verify_password("wonderland", "$2b$12$...")  # Utilisez un hash valide

def test_authenticate_user_success():
    user = authenticate_user({"alice": {"username": "alice", "hashed_password": "...", "disabled": False}}, "alice", "wonderland")
    assert user is not False

def test_authenticate_user_failure():
    user = authenticate_user({"alice": {"username": "alice", "hashed_password": "...", "disabled": False}}, "alice", "wrongpassword")
    assert user is False
