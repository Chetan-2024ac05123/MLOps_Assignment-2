"""Unit tests for FastAPI endpoints."""
import io
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked model."""
    with patch("api.app.get_model") as mock_get:
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([[0.85]]))
        mock_get.return_value = mock_model

        import api.app as app_module
        app_module._model = mock_model
        yield TestClient(app_module.app)
        app_module._model = None


@pytest.fixture
def dummy_jpeg():
    """Create dummy JPEG file bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_has_status_field(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_shows_model_loaded(self, client):
        response = client.get("/health")
        assert response.json()["model_loaded"] is True


class TestPredictEndpoint:
    def test_valid_image_returns_200(self, client, dummy_jpeg):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_has_prediction(self, client, dummy_jpeg):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")},
        )
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["cat", "dog"]

    def test_response_has_confidence(self, client, dummy_jpeg):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")},
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_has_filename(self, client, dummy_jpeg):
        response = client.post(
            "/predict",
            files={"file": ("my_cat.jpg", dummy_jpeg, "image/jpeg")},
        )
        assert response.json()["filename"] == "my_cat.jpg"


class TestModelNotLoaded:
    def test_predict_returns_503(self):
        import api.app as app_module
        app_module._model = None
        client = TestClient(app_module.app)
        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert response.status_code == 503


class TestAppMetrics:
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics/app")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "avg_latency_ms" in data
