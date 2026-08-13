"""
train.py
CNN model training with MLflow experiment tracking.
Builds a 4-layer CNN, trains with data augmentation, logs everything to MLflow.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
import mlflow  # noqa: E402

from src.preprocess import load_dataset_arrays  # noqa: E402

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
DROPOUT_RATE = 0.5
INPUT_SHAPE = (224, 224, 3)


def build_cnn_model(input_shape=INPUT_SHAPE):
    """Build a simple CNN for binary classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # Conv blocks
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        # Classification head
        layers.Flatten(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history, save_path):
    """Save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Train Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def train():
    """Full training pipeline with MLflow tracking."""
    print("=" * 50)
    print("  Cats vs Dogs — CNN Training")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow_uri = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cats_dogs_classification")
    print(f"\n[MLflow] Tracking URI: {mlflow_uri}")
    print("[MLflow] Experiment: cats_dogs_classification")

    # Load data
    print("\n[Step 1] Loading data...")
    X_train, y_train = load_dataset_arrays("train")
    X_val, y_val = load_dataset_arrays("val")
    X_test, y_test = load_dataset_arrays("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    with mlflow.start_run(run_name="cnn_baseline"):
        # Log parameters
        mlflow.log_params({
            "model_type": "CNN",
            "input_shape": str(INPUT_SHAPE),
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "augmentation": "flip+rotation+zoom",
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
        })

        # Build model
        print("\n[Step 2] Building CNN model...")
        model = build_cnn_model()
        model.summary()

        # Train
        print(f"\n[Step 3] Training for {EPOCHS} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        # Log per-epoch metrics
        for epoch in range(EPOCHS):
            mlflow.log_metrics({
                "train_loss": history.history["loss"][epoch],
                "train_accuracy": history.history["accuracy"][epoch],
                "val_loss": history.history["val_loss"][epoch],
                "val_accuracy": history.history["val_accuracy"][epoch],
            }, step=epoch + 1)

        # Evaluate
        print("\n[Step 4] Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_probs = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)

        report = classification_report(y_test, y_pred, target_names=["Cat", "Dog"], output_dict=True)
        print(f"\n  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

        # Log test metrics
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "test_precision": report["weighted avg"]["precision"],
            "test_recall": report["weighted avg"]["recall"],
            "test_f1": report["weighted avg"]["f1-score"],
        })

        # Save artifacts
        print("[Step 5] Saving artifacts...")
        curves_path = ARTIFACTS_DIR / "training_curves.png"
        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        plot_training_curves(history, curves_path)
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(cm_path))

        # Save model
        model_path = MODELS_DIR / "cats_dogs_cnn.h5"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))
        print(f"  Model saved: {model_path}")

        # Save metadata
        metadata = {
            "model_type": "CNN",
            "input_shape": list(INPUT_SHAPE),
            "classes": ["cat", "dog"],
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "epochs_trained": EPOCHS,
            "total_params": int(model.count_params()),
        }
        meta_path = MODELS_DIR / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(meta_path))

    print("\n" + "=" * 50)
    print("  Training Complete!")
    print(f"  Model: {model_path}")
    print(f"  MLflow: mlflow ui --backend-store-uri {mlflow_uri}")
    print("=" * 50)


if __name__ == "__main__":
    train()
