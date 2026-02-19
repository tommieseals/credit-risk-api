"""API tests."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample():
    return {"age": 35, "income": 75000, "employment_length": 5, "credit_score": 720,
            "num_credit_lines": 4, "credit_utilization": 0.3, "loan_amount": 25000,
            "loan_purpose": "debt_consolidation", "debt_to_income": 0.25, "num_delinquencies": 0}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


@patch("app.main.model_loader")
def test_predict(mock, client, sample):
    mock.is_loaded = True
    mock.version = "1.0.0"
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock.get_model.return_value = m
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert "probability_of_default" in r.json()


def test_invalid_input(client):
    assert client.post("/predict", json={"age": 15}).status_code == 422
