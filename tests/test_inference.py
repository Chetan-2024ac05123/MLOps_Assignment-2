"""Unit tests for model inference functions."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image
import io


@pytest.fixture
def dummy_image_bytes():
    """Create dummy JPEG bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_model():
    """Create a mock Keras model that returns a probability."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([[0.85]]))
    return model


class TestPreprocessImage:
    def test_output_shape(self, dummy_image_bytes):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_bytes)
        assert result.shape == (1, 224, 224, 3)

    def test_output_range(self, dummy_image_bytes):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_bytes)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_batch_dimension(self, dummy_image_bytes):
        from src.predict import preprocess_image
        result = preprocess_image(dummy_image_bytes)
        assert result.ndim == 4


class TestPredict:
    def test_returns_expected_keys(self, mock_model, dummy_image_bytes):
        from src.predict import preprocess_image, predict
        arr = preprocess_image(dummy_image_bytes)
        result = predict(mock_model, arr)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_dog_prediction(self, mock_model, dummy_image_bytes):
        from src.predict import preprocess_image, predict
        mock_model.predict.return_value = np.array([[0.9]])
        arr = preprocess_image(dummy_image_bytes)
        result = predict(mock_model, arr)
        assert result["prediction"] == "dog"
        assert result["confidence"] == 0.9

    def test_cat_prediction(self, mock_model, dummy_image_bytes):
        from src.predict import preprocess_image, predict
        mock_model.predict.return_value = np.array([[0.2]])
        arr = preprocess_image(dummy_image_bytes)
        result = predict(mock_model, arr)
        assert result["prediction"] == "cat"
        assert result["confidence"] == 0.8

    def test_probabilities_sum_to_one(self, mock_model, dummy_image_bytes):
        from src.predict import preprocess_image, predict
        arr = preprocess_image(dummy_image_bytes)
        result = predict(mock_model, arr)
        total = result["probabilities"]["cat"] + result["probabilities"]["dog"]
        assert abs(total - 1.0) < 0.01

    def test_confidence_between_0_and_1(self, mock_model, dummy_image_bytes):
        from src.predict import preprocess_image, predict
        for prob in [0.1, 0.3, 0.5, 0.7, 0.99]:
            mock_model.predict.return_value = np.array([[prob]])
            arr = preprocess_image(dummy_image_bytes)
            result = predict(mock_model, arr)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_model_not_found_raises(self):
        from src.predict import get_model
        import src.predict as pred_module
        pred_module._model = None
        with pytest.raises(FileNotFoundError):
            get_model("/nonexistent/model.h5")
