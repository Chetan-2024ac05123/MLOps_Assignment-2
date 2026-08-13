"""
smoke_test.py
Post-deployment smoke test: health check + prediction.
"""
import sys
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def test_health():
    """Check /health endpoint."""
    resp = requests.get(f"{API_URL}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "healthy", f"Unhealthy: {data}"
    assert data["model_loaded"] is True, "Model not loaded"
    print(f"  Health: OK — {data}")


def test_predict():
    """Send a test image to /predict."""
    # Create a minimal test image
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        f"{API_URL}/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        timeout=30,
    )
    assert resp.status_code == 200, f"Predict failed: {resp.status_code}"
    data = resp.json()
    assert "prediction" in data, f"No prediction in response: {data}"
    assert data["prediction"] in ["cat", "dog"], f"Invalid prediction: {data}"
    print(f"  Predict: OK — {data['prediction']} ({data['confidence']:.2%})")


def main():
    print("=" * 50)
    print("  Smoke Test")
    print("=" * 50)
    try:
        test_health()
        test_predict()
        print("\n  All smoke tests PASSED")
        return 0
    except Exception as e:
        print(f"\n  FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
