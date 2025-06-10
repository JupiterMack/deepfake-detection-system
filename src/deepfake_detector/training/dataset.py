# src/deepfake_detector/training/dataset.py

"""
Defines a custom PyTorch Dataset for loading deepfake image data.

This module provides the DeepfakeDataset class, which is designed to work with
a dataset structured with a metadata CSV file. The CSV file should map image
file paths to their corresponding labels (e.g., 0 for 'real', 1 for 'fake').
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

logger = logging.getLogger(__name__)

# Define a type alias for clarity
TransformType = Optional[Callable[[Image.Image], torch.Tensor]]


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading deepfake images.

    This dataset class reads a CSV file containing image paths and their labels.
    It loads images, applies specified transformations, and returns image-label
    pairs for training or evaluation.

    Args:
        metadata_file (str or Path): Path to the CSV file containing image metadata.
            The CSV must have 'path' and 'label' columns.
        dataset_dir (str or Path): The root directory of the dataset. Image paths
            in the metadata file are considered relative to this directory.
        transform (TransformType, optional): A function/transform that takes in a
            PIL image and returns a transformed version. Defaults to None.
    """

    def __init__(self, metadata_file: str | Path, dataset_dir: str | Path, transform: TransformType = None):
        """
        Initializes the dataset.
        """
        self.dataset_dir = Path(dataset_dir)
        metadata_path = Path(metadata_file)

        if not metadata_path.is_file():
            logger.error(f"Metadata file not found at: {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        if not self.dataset_dir.is_dir():
            logger.error(f"Dataset directory not found at: {self.dataset_dir}")
            raise FileNotFoundError(f"Dataset directory not found at: {self.dataset_dir}")

        # Load the metadata from the CSV file
        try:
            self.metadata = pd.read_csv(metadata_path)
            # Validate required columns
            if 'path' not in self.metadata.columns or 'label' not in self.metadata.columns:
                raise ValueError("Metadata CSV must contain 'path' and 'label' columns.")
        except Exception as e:
            logger.error(f"Error reading or parsing metadata file {metadata_path}: {e}")
            raise IOError(f"Error reading or parsing metadata file {metadata_path}: {e}") from e

        self.transform = transform
        logger.info(f"Initialized dataset with {len(self.metadata)} samples from {metadata_path}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed
            image tensor and the corresponding label tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and label from metadata
        img_relative_path = self.metadata.iloc[idx, self.metadata.columns.get_loc('path')]
        label = self.metadata.iloc[idx, self.metadata.columns.get_loc('label')]

        # Construct the full image path
        img_path = self.dataset_dir / img_relative_path

        try:
            # Load image using Pillow and ensure it's in RGB format
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Could not load or identify image at path: {img_path}. Error: {e}")
            # To prevent training from crashing, we can return a dummy tensor.
            # The collate function in the DataLoader should be able to handle this if needed.
            # For simplicity here, we re-raise, assuming the dataset is clean.
            raise IOError(f"Error opening or identifying image file {img_path}") from e

        # Apply transformations if they exist
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default transformation if none is provided
            image_tensor = transforms.ToTensor()(image)

        # Convert label to a tensor.
        # Using FloatTensor for labels is common for BCEWithLogitsLoss.
        label_tensor = torch.tensor(label, dtype=torch.float32)
        # Add a dimension to match model output shape (batch_size, 1)
        label_tensor = label_tensor.unsqueeze(0)

        return image_tensor, label_tensor


def get_default_transforms(image_size: int = 224, is_train: bool = True) -> Callable:
    """
    Provides a default set of transformations for the dataset.

    These transformations include resizing, conversion to tensor, and normalization.
    For training, it also includes common data augmentation techniques.

    Args:
        image_size (int): The target size for the images (height and width).
        is_train (bool): If True, applies data augmentation suitable for training.

    Returns:
        Callable: A composed torchvision transform.
    """
    # Normalization constants for models pre-trained on ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        # Augmentation and preprocessing for the training set
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    else:
        # Preprocessing for the validation/test set (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]

    return transforms.Compose(transform_list)
```