"""Unit tests for image preprocessing functions."""
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample RGB JPEG image."""
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


@pytest.fixture
def grayscale_image_path(tmp_path):
    """Create a grayscale JPEG image."""
    img = Image.fromarray(np.random.randint(0, 255, (200, 200), dtype=np.uint8), mode="L")
    path = tmp_path / "gray_image.jpg"
    img.save(path)
    return str(path)


class TestLoadAndResize:
    def test_output_shape(self, sample_image_path):
        from src.preprocess import load_and_resize_image
        result = load_and_resize_image(sample_image_path)
        assert result.shape == (224, 224, 3)

    def test_pixel_values_normalised(self, sample_image_path):
        from src.preprocess import load_and_resize_image
        result = load_and_resize_image(sample_image_path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_float32(self, sample_image_path):
        from src.preprocess import load_and_resize_image
        result = load_and_resize_image(sample_image_path)
        assert result.dtype == np.float32

    def test_grayscale_converted_to_rgb(self, grayscale_image_path):
        from src.preprocess import load_and_resize_image
        result = load_and_resize_image(grayscale_image_path)
        assert result.shape == (224, 224, 3)

    def test_custom_target_size(self, sample_image_path):
        from src.preprocess import load_and_resize_image
        result = load_and_resize_image(sample_image_path, target_size=(128, 128))
        assert result.shape == (128, 128, 3)

    def test_non_square_resize(self, sample_image_path):
        from src.preprocess import load_and_resize_image
        # PIL.resize takes (width, height); numpy shape is (height, width, channels)
        result = load_and_resize_image(sample_image_path, target_size=(100, 200))
        assert result.shape == (200, 100, 3)


class TestCollectImages:
    def test_collects_from_flat_dir(self, tmp_path):
        from src.preprocess import collect_images
        for i in range(3):
            Image.new("RGB", (10, 10)).save(tmp_path / f"cat.{i}.jpg")
            Image.new("RGB", (10, 10)).save(tmp_path / f"dog.{i}.jpg")

        cats, dogs = collect_images(tmp_path)
        assert len(cats) == 3
        assert len(dogs) == 3

    def test_collects_from_subdirs(self, tmp_path):
        from src.preprocess import collect_images
        (tmp_path / "cats").mkdir()
        (tmp_path / "dogs").mkdir()
        for i in range(4):
            Image.new("RGB", (10, 10)).save(tmp_path / "cats" / f"img{i}.jpg")
            Image.new("RGB", (10, 10)).save(tmp_path / "dogs" / f"img{i}.jpg")

        cats, dogs = collect_images(tmp_path)
        assert len(cats) == 4
        assert len(dogs) == 4

    def test_empty_dir_returns_empty(self, tmp_path):
        from src.preprocess import collect_images
        cats, dogs = collect_images(tmp_path)
        assert len(cats) == 0
        assert len(dogs) == 0
