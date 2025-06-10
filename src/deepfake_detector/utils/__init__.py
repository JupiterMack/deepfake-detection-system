# src/deepfake_detector/utils/__init__.py

"""
Utilities Package for the Deepfake Detection System.

This package provides a collection of helper functions and classes used across
the application, but not specific to any single component like models or data
processing. This includes utilities for:

- Reproducibility: Functions to set random seeds for consistent results.
- Training Metrics: Classes to track metrics like loss and accuracy during training.
- Checkpoint Management: Helpers for saving and loading model and optimizer states.
- Visualization: Functions to create visual outputs, such as overlaying heatmaps.
- General Helpers: Other miscellaneous helper functions.

The main components are exposed at the top level of this package for easy access.
"""

# The following imports assume that the actual utility functions are organized
# into separate modules within this 'utils' package for better organization.
# For example:
# - src/deepfake_detector/utils/reproducibility.py contains set_seed
# - src/deepfake_detector/utils/metrics.py contains AverageMeter
# - src/deepfake_detector/utils/checkpoint.py contains save/load checkpoint functions
# - src/deepfake_detector/utils/visualization.py contains visualization helpers

# If this package remains small, these functions could also be defined directly
# in a single file like 'helpers.py' and imported from there.

from .reproducibility import set_seed
from .metrics import AverageMeter
from .checkpoint import save_checkpoint, load_checkpoint
from .visualization import overlay_heatmap_on_image

__all__ = [
    "set_seed",
    "AverageMeter",
    "save_checkpoint",
    "load_checkpoint",
    "overlay_heatmap_on_image",
]
```