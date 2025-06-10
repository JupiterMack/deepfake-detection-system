# tests/processing/test_image_loader.py

"""
Unit tests for the image loading and preprocessing functions.

This module tests the functions in `src.deepfake_detector.processing.image_loader`
to ensure they correctly load images from different sources (path, bytes) and
preprocess them into the tensor format required by the model.
"""

import pytest
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
import torchvision.transforms as T

from src.deepfake_detector.processing.image_loader import (
    load_image,
    preprocess_image,
    get_image_transform,
)

# --- Fixtures ---

@pytest.fixture(scope="module")
def dummy_image_path(tmp_path_factory) -> Path:
    """
    Creates a dummy PNG image file in a temporary directory and returns its path.
    The fixture has a 'module' scope, so the image is created only once per test module.
    """
    img_dir = tmp_path_factory.mktemp("images")
    img_path = img_dir / "test_image.png"
    # Create a 100x150 RGB image with random pixel values
    img_array = np.random.randint(0, 256, (150, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, "RGB")
    img.save(img_path)
    return img_path

@pytest.fixture(scope="module")
def dummy_image_bytes(dummy_image_path) -> BytesIO:
    """
    Provides the byte content of the dummy image in a BytesIO stream.
    """
    with open(dummy_image_path, "rb") as f:
        byte_io = BytesIO(f.read())
    byte_io.seek(0)  # Reset stream pointer to the beginning for reading
    return byte_io

@pytest.fixture(scope="module")
def dummy_pil_image(dummy_image_path) -> Image.Image:
    """
    Provides a dummy PIL.Image object.
    """
    return Image.open(dummy_image_path)


# --- Test Cases ---

def test_load_image_from_path(dummy_image_path):
    """
    Tests loading an image from a valid pathlib.Path object.
    """
    image = load_image(dummy_image_path)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"  # The function should convert to RGB
    assert image.size == (100, 150)

def test_load_image_from_path_string(dummy_image_path):
    """
    Tests loading an image from a valid file path provided as a string.
    """
    image = load_image(str(dummy_image_path))
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (100, 150)

def test_load_image_from_bytes(dummy_image_bytes):
    """
    Tests loading an image from an in-memory byte stream (BytesIO).
    """
    image = load_image(dummy_image_bytes)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (100, 150)

def test_load_image_nonexistent_path():
    """
    Tests that loading from a non-existent path raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_image(Path("non/existent/path/for/testing.jpg"))

def test_load_image_invalid_bytes():
    """
    Tests that loading from invalid byte data raises an UnidentifiedImageError from Pillow.
    """
    invalid_bytes = BytesIO(b"this is definitely not a valid image file")
    with pytest.raises(UnidentifiedImageError):
        load_image(invalid_bytes)

def test_get_image_transform():
    """
    Tests the creation of the image transformation pipeline.
    """
    target_size = 224
    transform = get_image_transform(image_size=target_size)

    # Check that it's a Compose object from torchvision
    assert isinstance(transform, T.Compose)

    # Check that it contains the expected sequence of transforms
    transform_classes = [t.__class__ for t in transform.transforms]
    assert transform_classes[0] == T.Resize
    assert transform_classes[1] == T.ToTensor
    assert transform_classes[2] == T.Normalize

    # Verify the parameters of the Resize transform
    resize_transform = transform.transforms[0]
    assert resize_transform.size == [target_size, target_size]

def test_preprocess_image_from_pil(dummy_pil_image):
    """
    Tests the full preprocessing pipeline on a given PIL.Image object.
    """
    target_size = 256
    tensor = preprocess_image(dummy_pil_image, image_size=target_size)

    # 1. Check type: should be a torch.Tensor
    assert isinstance(tensor, torch.Tensor)

    # 2. Check shape: should be [batch_size, channels, height, width]
    expected_shape = (1, 3, target_size, target_size)
    assert tensor.shape == expected_shape

    # 3. Check data type: should be float32 for model input
    assert tensor.dtype == torch.float32

    # 4. Check value range: after normalization, values should not be in 0-255 or 0-1
    assert tensor.min() < 0.0
    assert tensor.max() < 2.0  # Normalized values are typically in a small range around 0

def test_preprocess_image_from_path(dummy_image_path):
    """
    Tests the preprocessing function when given a file path directly.
    It should handle loading and transforming in one call.
    """
    target_size = 224
    tensor = preprocess_image(dummy_image_path, image_size=target_size)

    assert isinstance(tensor, torch.Tensor)
    expected_shape = (1, 3, target_size, target_size)
    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float32

def test_preprocess_image_from_bytes(dummy_image_bytes):
    """
    Tests the preprocessing function when given image bytes directly.
    """
    target_size = 299  # Test with a different common size
    tensor = preprocess_image(dummy_image_bytes, image_size=target_size)

    assert isinstance(tensor, torch.Tensor)
    expected_shape = (1, 3, target_size, target_size)
    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float32

def test_preprocess_image_invalid_input_type():
    """
    Tests that preprocess_image raises a TypeError for invalid input types
    that are not a path, bytes, or PIL Image.
    """
    with pytest.raises(TypeError):
        preprocess_image(12345)  # Pass an integer

    with pytest.raises(TypeError):
        preprocess_image([1, 2, 3]) # Pass a list
```