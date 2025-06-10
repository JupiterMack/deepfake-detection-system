# src/deepfake_detector/training/__init__.py

"""
Training Sub-package for the Deepfake Detection System.

This package contains all the components related to the model training process,
including the main training loop, evaluation logic, loss functions, and other
training-specific utilities.

The central component is the `Trainer` class, which orchestrates the entire
training and validation pipeline.
"""

# Expose the main Trainer class for easy access from other parts of the application,
# particularly from the training script.
from .trainer import Trainer

# Define the public API of the 'training' package.
__all__ = [
    "Trainer",
]