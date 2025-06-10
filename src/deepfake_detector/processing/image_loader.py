# src/deepfake_detector/processing/image_loader.py

"""
Functions for loading, preprocessing, and transforming images for the deepfake detection model.

This module provides a standardized way to prepare image data, whether it comes from a
file path or an in-memory byte stream, making it suitable for model inference.
"""

import io
from pathlib import Path
from typing import Union, Tuple

import torch
from PIL import Image
from torchvision import transforms

# --- Constants ---
# Default image size expected by many pre-trained models.
# This should match the size the model was trained on.
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Standard normalization values for models pre-trained on ImageNet.
# Using these is a common practice as they work well with many CNN architectures.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def get_preprocessing_transform(
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD
) -> transforms.Compose:
    """
    Creates a torchvision transform pipeline for preprocessing images.

    This is the standard pipeline used for inference.

    Args:
        size (Tuple[int, int]): The target size (height, width) to resize the image to.
        mean (list[float]): The mean values for normalization.
        std (list[float]): The standard deviation values for normalization.

    Returns:
        transforms.Compose: A composition of torchvision transforms.
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def load_and_preprocess_image(
    image_source: Union[str, Path, bytes, io.BytesIO],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Loads an image from a path or byte stream, preprocesses it, and returns a tensor.

    The preprocessing steps include:
    1. Opening the image using Pillow.
    2. Converting to RGB format to ensure channel consistency.
    3. Resizing to the target size.
    4. Converting to a PyTorch tensor (scales pixel values to [0, 1]).
    5. Normalizing with ImageNet mean and std.
    6. Adding a batch dimension to create a 4D tensor [1, C, H, W].
    7. Moving the tensor to the specified device (e.g., 'cpu' or 'cuda').

    Args:
        image_source (Union[str, Path, bytes, io.BytesIO]): The source of the image.
            Can be a file path or an in-memory byte stream.
        target_size (Tuple[int, int]): The target size (height, width) for the image.
        device (torch.device): The device to move the final tensor to.

    Returns:
        torch.Tensor: The preprocessed image tensor ready for model input.

    Raises:
        IOError: If the image_source cannot be opened or identified as an image file.
    """
    if isinstance(image_source, bytes):
        image_source = io.BytesIO(image_source)

    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(image_source).convert("RGB")
    except Exception as e:
        raise IOError(f"Failed to open or process image from source. Error: {e}")

    # Get the standard transformation pipeline
    preprocess = get_preprocessing_transform(size=target_size)

    # Apply transformations
    tensor = preprocess(image)

    # Add batch dimension (B, C, H, W) and move to the target device
    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses the normalization on a tensor to make it viewable as an image.

    This is useful for visualizing tensors, for example, when displaying
    explainable AI (XAI) heatmaps on top of the original image.

    Assumes the tensor was normalized with IMAGENET_MEAN and IMAGENET_STD.
    The input tensor can be a single image (C, H, W) or a batch (B, C, H, W).

    Args:
        tensor (torch.Tensor): The normalized image tensor.

    Returns:
        torch.Tensor: The denormalized image tensor with pixel values clamped to [0, 1].
    """
    # Create tensors for mean and std, and reshape them to be broadcastable
    # The shape should be (1, C, 1, 1) to work with a batch of images (B, C, H, W)
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)

    # If the input tensor is a single image (C, H, W), add a batch dimension for broadcasting
    is_single_image = tensor.dim() == 3
    if is_single_image:
        tensor = tensor.unsqueeze(0)

    # Denormalize: (tensor * std) + mean
    denormalized_tensor = tensor * std + mean

    # If a batch dimension was added, remove it
    if is_single_image:
        denormalized_tensor = denormalized_tensor.squeeze(0)

    # Clamp the values to be in the valid [0, 1] range for image display
    return denormalized_tensor.clamp(0, 1)
```