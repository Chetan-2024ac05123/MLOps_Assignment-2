"""
run_pipeline.py
Run the complete ML pipeline: data loading, preprocessing, training, testing.
"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name, command):
    """Run a pipeline step and check for errors."""
    print(f"\n{'=' * 60}")
    print(f"  Step: {step_name}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        command, shell=True, cwd=BASE_DIR,
        env={**os.environ, "PYTHONPATH": BASE_DIR}
    )
    if result.returncode != 0:
        print(f"\n  FAILED: {step_name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  DONE: {step_name}")


def main():
    print("=" * 60)
    print("  Cats vs Dogs — Full Pipeline")
    print("=" * 60)

    steps = [
        ("1. Data Loading", f"{sys.executable} -m src.data_loader"),
        ("2. Preprocessing", f"{sys.executable} -m src.preprocess"),
        ("3. Model Training", f"{sys.executable} -m src.train"),
        ("4. Unit Tests", f"{sys.executable} -m pytest tests/ -v"),
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  docker build -t cats-dogs-api:latest .")
    print("  docker run -d -p 8000:8000 cats-dogs-api:latest")


if __name__ == "__main__":
    main()
