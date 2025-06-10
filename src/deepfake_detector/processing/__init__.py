"""
Processing Sub-package for the Deepfake Detection System.

This package contains modules for data handling, including:
- Preprocessing: Functions for resizing, normalizing, and converting images to tensors.
- Datasets: Custom PyTorch Dataset classes for loading and serving image data for training and evaluation.
- Augmentations: Image transformations for training data augmentation to improve model robustness.

The main components are exposed at the package level for convenient access from other parts of the application,
such as the training script or the inference pipeline.
"""

# The following imports assume the existence of `preprocessing.py` and `dataset.py`
# modules within this 'processing' package. These modules would contain the actual
# implementation of the functions and classes. This __init__.py file makes them
# accessible at the package level.

# For example, from a hypothetical `preprocessing.py`:
# from .preprocessing import preprocess_image, get_default_transforms

# And from a hypothetical `dataset.py`:
# from .dataset import DeepfakeDataset

# By exposing them here, other parts of the application can use:
# from deepfake_detector.processing import preprocess_image
# instead of the more verbose:
# from deepfake_detector.processing.preprocessing import preprocess_image


# To make the package functional even if submodules are not yet created,
# we can use try-except blocks. However, for a clean initial structure,
# we will assume these will be created and define the public API via __all__.

__all__ = [
    "preprocess_image",
    "get_default_transforms",
    "DeepfakeDataset",
]

# Note: The actual implementation for the names listed in `__all__`
# would reside in other files within this `processing` directory,
# such as `preprocessing.py` and `dataset.py`. This `__init__.py`
# would then import them to expose them. For now, this file serves
# as a placeholder and defines the public API of the package.