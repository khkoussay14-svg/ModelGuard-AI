
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Create a test client and initialize the ML model."""
    with TestClient(app) as test_client:
        yield test_client


def test_home_endpoint(client):
    """Verify that the API home endpoint is accessible."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to ModelGuard AI"
    }


def test_valid_prediction(client):
    """Verify that valid features produce a prediction."""

    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"


def test_negative_feature(client):
    """Verify that negative features are rejected."""

    response = client.post(
        "/predict",
        json={"features": [5.1, -3.5, 1.4, 0.2]}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Features must contain four positive numbers."
    )


def test_missing_feature(client):
    """Verify that incomplete feature lists are rejected."""

    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4]}
    )

    assert response.status_code == 422

def test_compare_endpoint(client):
    """Verify that the comparison API returns valid model results."""

    # Send an HTTP GET request to the comparison endpoint
    response = client.get("/compare")

    # Check whether the request was successful
    assert response.status_code == 200

    # Parse the JSON response into a Python dictionary
    data = response.json()

    # Verify that the response contains the expected key
    assert "models" in data

    # Extract the model evaluation results
    models = data["models"]

    # Verify that exactly three models were evaluated
    assert len(models) == 3

    # Check all model metrics
    for model in models:
        assert "model" in model
        assert 0.0 <= model["accuracy"] <= 1.0
        assert 0.0 <= model["precision"] <= 1.0
        assert 0.0 <= model["recall"] <= 1.0
        assert 0.0 <= model["f1_score"] <= 1.0

