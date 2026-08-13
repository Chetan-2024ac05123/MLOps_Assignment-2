"""
data_loader.py
Download and organise the Cats vs Dogs dataset.
Supports Kaggle CLI, local copy, or manual download instructions.
"""
import os
import shutil
import zipfile
import random
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = BASE_DIR / "data" / "raw"

SEED = 42
IMAGES_PER_CLASS = 500  # subset for faster training


def organise_kaggle_extract(extract_dir: Path, output_dir: Path, n_per_class: int = IMAGES_PER_CLASS):
    """Organise extracted Kaggle images into cats/ and dogs/ subfolders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cats_dir = output_dir / "cats"
    dogs_dir = output_dir / "dogs"
    cats_dir.mkdir(exist_ok=True)
    dogs_dir.mkdir(exist_ok=True)

    # Kaggle dataset has cat.0.jpg ... cat.N.jpg and dog.0.jpg ... dog.N.jpg
    # They may be in a 'train' subfolder
    search_dirs = [extract_dir]
    for sub in ["train", "Train", "PetImages"]:
        if (extract_dir / sub).exists():
            search_dirs.append(extract_dir / sub)

    cat_imgs = []
    dog_imgs = []
    for d in search_dirs:
        cat_imgs.extend(sorted(d.glob("cat*.jpg")))
        dog_imgs.extend(sorted(d.glob("dog*.jpg")))
        if (d / "Cat").exists():
            cat_imgs.extend(sorted((d / "Cat").glob("*.jpg")))
        if (d / "Dog").exists():
            dog_imgs.extend(sorted((d / "Dog").glob("*.jpg")))

    random.seed(SEED)
    random.shuffle(cat_imgs)
    random.shuffle(dog_imgs)

    cat_imgs = cat_imgs[:n_per_class]
    dog_imgs = dog_imgs[:n_per_class]

    print(f"Copying {len(cat_imgs)} cat images to {cats_dir}")
    for img in cat_imgs:
        shutil.copy2(img, cats_dir / img.name)

    print(f"Copying {len(dog_imgs)} dog images to {dogs_dir}")
    for img in dog_imgs:
        shutil.copy2(img, dogs_dir / img.name)

    return len(cat_imgs), len(dog_imgs)


def download_kaggle(output_dir: Path = RAW_DIR):
    """Download Cats vs Dogs dataset via Kaggle CLI."""
    import subprocess
    tmp_dir = BASE_DIR / "data" / "_tmp_kaggle"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset via Kaggle CLI...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "dogs-vs-cats", "-p", str(tmp_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Try the alternative dataset
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "tongpython/cat-and-dog", "-p", str(tmp_dir)],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    # Extract zip
    for zf in tmp_dir.glob("*.zip"):
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(tmp_dir / "extracted")

    n_cats, n_dogs = organise_kaggle_extract(tmp_dir / "extracted", output_dir)
    shutil.rmtree(tmp_dir)
    return n_cats, n_dogs


def copy_from_sibling(output_dir: Path = RAW_DIR):
    """Copy data from existing cats-dogs-mlops project if available."""
    sibling = BASE_DIR.parent / "cats-dogs-mlops" / "data" / "raw"
    if not sibling.exists():
        return False

    cat_imgs = sorted(sibling.glob("cat*.jpg"))
    dog_imgs = sorted(sibling.glob("dog*.jpg"))

    if not cat_imgs and (sibling / "cats").exists():
        cat_imgs = sorted((sibling / "cats").glob("*.jpg"))
    if not dog_imgs and (sibling / "dogs").exists():
        dog_imgs = sorted((sibling / "dogs").glob("*.jpg"))

    if len(cat_imgs) == 0 or len(dog_imgs) == 0:
        return False

    print(f"Found {len(cat_imgs)} cats, {len(dog_imgs)} dogs in sibling project")
    cats_dir = output_dir / "cats"
    dogs_dir = output_dir / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    random.shuffle(cat_imgs)
    random.shuffle(dog_imgs)

    for img in cat_imgs[:IMAGES_PER_CLASS]:
        dest = cats_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)

    for img in dog_imgs[:IMAGES_PER_CLASS]:
        dest = dogs_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)

    return True


def check_existing(raw_dir: Path = RAW_DIR):
    """Check if data already exists."""
    cats = list((raw_dir / "cats").glob("*.jpg")) if (raw_dir / "cats").exists() else []
    dogs = list((raw_dir / "dogs").glob("*.jpg")) if (raw_dir / "dogs").exists() else []
    # Also check flat structure
    if not cats:
        cats = list(raw_dir.glob("cat*.jpg"))
    if not dogs:
        dogs = list(raw_dir.glob("dog*.jpg"))
    return len(cats), len(dogs)


def load_dataset(raw_dir: Path = RAW_DIR):
    """Main entry point: ensure dataset is available."""
    n_cats, n_dogs = check_existing(raw_dir)
    if n_cats >= 100 and n_dogs >= 100:
        print(f"Dataset already present: {n_cats} cats, {n_dogs} dogs")
        return n_cats, n_dogs

    print("Dataset not found. Attempting to locate data...")

    # Try sibling project first
    if copy_from_sibling(raw_dir):
        n_cats, n_dogs = check_existing(raw_dir)
        print(f"Copied from sibling project: {n_cats} cats, {n_dogs} dogs")
        return n_cats, n_dogs

    # Try Kaggle CLI
    try:
        n_cats, n_dogs = download_kaggle(raw_dir)
        print(f"Downloaded via Kaggle: {n_cats} cats, {n_dogs} dogs")
        return n_cats, n_dogs
    except Exception as e:
        print(f"Kaggle download failed: {e}")

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("1. Go to: https://www.kaggle.com/c/dogs-vs-cats/data")
    print("2. Download 'train.zip'")
    print(f"3. Extract cat/dog images into: {raw_dir}/cats/ and {raw_dir}/dogs/")
    print("   OR place cat.0.jpg ... cat.N.jpg and dog.0.jpg ... dog.N.jpg")
    print(f"   directly in: {raw_dir}/")
    print("4. Re-run: python -m src.data_loader")
    print("=" * 60)
    raise FileNotFoundError("Dataset not available. See instructions above.")


if __name__ == "__main__":
    print("=" * 50)
    print("  Cats vs Dogs — Data Loader")
    print("=" * 50)
    n_cats, n_dogs = load_dataset()
    print(f"\nDataset ready: {n_cats} cats, {n_dogs} dogs")
    print(f"Location: {RAW_DIR}")
