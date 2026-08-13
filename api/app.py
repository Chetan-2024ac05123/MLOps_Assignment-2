"""
FastAPI inference service for Cats vs Dogs classification.
Endpoints: /health, /predict, /metrics, /metrics/app
"""
import time
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.predict import get_model, preprocess_image, predict

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cats_dogs_api")

app = FastAPI(
    title="Cats vs Dogs API",
    description="Binary image classification — predict cat or dog from an uploaded image.",
    version="1.0.0",
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# App-level counters
_stats = {"total_predictions": 0, "total_latency_ms": 0.0}

# Load model at startup
_model = None


@app.on_event("startup")
def startup():
    global _model
    try:
        _model = get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration_ms:.1f}ms"
    )
    return response


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        if not (file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))):
            raise HTTPException(status_code=422, detail="File must be an image (JPEG, PNG, etc.)")

    start = time.time()
    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        result = predict(_model, image_array)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms = (time.time() - start) * 1000
    _stats["total_predictions"] += 1
    _stats["total_latency_ms"] += latency_ms

    logger.info(
        f"Prediction: {result['prediction']} "
        f"confidence={result['confidence']:.4f} "
        f"file={file.filename} latency={latency_ms:.1f}ms"
    )

    return {
        "filename": file.filename,
        **result,
    }


@app.get("/metrics/app")
def app_metrics():
    avg_latency = (
        _stats["total_latency_ms"] / _stats["total_predictions"]
        if _stats["total_predictions"] > 0
        else 0.0
    )
    return {
        "total_predictions": _stats["total_predictions"],
        "avg_latency_ms": round(avg_latency, 2),
    }
