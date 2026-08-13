"""
predict.py
Model inference utilities for Cats vs Dogs classification.
"""
import os
import io
import numpy as np
from PIL import Image
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "cats_dogs_cnn.h5"))
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]

_model = None


def get_model(model_path=None):
    """Load the trained CNN model (cached)."""
    global _model
    if _model is not None:
        return _model

    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    import tensorflow as tf
    _model = tf.keras.models.load_model(model_path)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes for model input."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image_array: np.ndarray) -> dict:
    """Run inference and return prediction with probabilities."""
    prob = float(model.predict(image_array, verbose=0)[0][0])
    predicted_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1.0 - prob

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(1.0 - prob, 4),
            "dog": round(prob, 4),
        },
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    model = get_model()
    arr = preprocess_image(img_bytes)
    result = predict(model, arr)
    print(f"File: {image_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probabilities: {result['probabilities']}")
