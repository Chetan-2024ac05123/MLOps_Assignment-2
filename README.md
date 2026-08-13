# Cats vs Dogs — MLOps Pipeline (Assignment 2)

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs)
using CNN, DVC, MLflow, Docker, Kubernetes, and CI/CD.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader          # download dataset
python -m src.preprocess           # split into train/val/test
python -m src.train                # train CNN + log to MLflow
pytest tests/ -v                   # run unit tests
uvicorn api.app:app --port 8000    # start API
```

## Project Structure

```
├── src/           # ML pipeline (data_loader, preprocess, train, predict)
├── api/           # FastAPI inference service
├── tests/         # Unit tests (preprocess, inference, API)
├── k8s/           # Kubernetes manifests
├── monitoring/    # Prometheus config
├── notebooks/     # EDA + end-to-end demo
├── report/        # Report, screenshots, video script
├── Dockerfile     # Container definition
├── dvc.yaml       # DVC pipeline stages
└── .github/       # CI/CD workflow
```

## Dataset

Cats and Dogs binary classification dataset from Kaggle.
Pre-processed to 224×224 RGB, split 80/10/10 (train/val/test).
