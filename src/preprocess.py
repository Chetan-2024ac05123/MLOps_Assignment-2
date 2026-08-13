"""
preprocess.py
Image preprocessing: resize to 224x224 RGB, normalise, split train/val/test.
"""
import os
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path

IMG_SIZE = (224, 224)
SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_and_resize_image(img_path, target_size=IMG_SIZE):
    """Load image, convert to RGB, resize, normalise to [0, 1]."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def collect_images(raw_dir=None):
    """Collect cat and dog image paths from raw directory."""
    if raw_dir is None:
        raw_dir = RAW_DIR
    raw_dir = Path(raw_dir)

    cat_imgs = sorted(raw_dir.glob("cat*.jpg"))
    dog_imgs = sorted(raw_dir.glob("dog*.jpg"))

    # Try subfolder structure
    if not cat_imgs and (raw_dir / "cats").exists():
        cat_imgs = sorted((raw_dir / "cats").glob("*.jpg"))
    if not dog_imgs and (raw_dir / "dogs").exists():
        dog_imgs = sorted((raw_dir / "dogs").glob("*.jpg"))

    return cat_imgs, dog_imgs


def split_dataset(raw_dir=None, output_dir=None, ratios=None):
    """Split raw images into train/val/test folders."""
    if raw_dir is None:
        raw_dir = RAW_DIR
    if output_dir is None:
        output_dir = PROCESSED_DIR
    if ratios is None:
        ratios = SPLIT_RATIOS

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    cat_imgs, dog_imgs = collect_images(raw_dir)
    print(f"Found {len(cat_imgs)} cat images, {len(dog_imgs)} dog images")

    if len(cat_imgs) == 0 or len(dog_imgs) == 0:
        raise ValueError(f"No images found in {raw_dir}. Run data_loader first.")

    random.seed(SEED)

    for label, images in [("cat", cat_imgs), ("dog", dog_imgs)]:
        imgs = list(images)
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dest = output_dir / split_name / label
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy2(img_path, dest / img_path.name)
            print(f"  {label}/{split_name}: {len(split_imgs)} images")

    print(f"Dataset split complete -> {output_dir}")


def load_dataset_arrays(split="train", processed_dir=None):
    """Load a split into numpy arrays (X, y)."""
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    processed_dir = Path(processed_dir)
    split_dir = processed_dir / split

    images, labels = [], []
    for label_idx, label_name in enumerate(["cat", "dog"]):
        label_dir = split_dir / label_name
        if not label_dir.exists():
            continue
        for img_path in sorted(label_dir.glob("*.jpg")):
            try:
                img = load_and_resize_image(img_path)
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")

    X = np.array(images)
    y = np.array(labels)
    print(f"Loaded {split}: {X.shape[0]} images, shape={X.shape[1:]}")
    return X, y


if __name__ == "__main__":
    print("=" * 50)
    print("  Cats vs Dogs — Preprocessing")
    print("=" * 50)
    split_dataset()
    print("\nVerifying splits...")
    for split in ["train", "val", "test"]:
        X, y = load_dataset_arrays(split)
        print(f"  {split}: {X.shape}, labels={np.bincount(y)}")
