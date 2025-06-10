# src/deepfake_detector/xai/__init__.py

"""
Explainable AI (XAI) Sub-package for the Deepfake Detection System.

This package provides the tools and methods to interpret the predictions of the
deepfake detection models. It leverages libraries like Captum to generate
attribution maps (saliency maps or heatmaps) that highlight which parts of an
image were most influential in the model's decision-making process.

The primary goal is to answer "Why did the model classify this image as real or fake?"
by visualizing the evidence it used.

This package is designed to be modular, allowing for the easy addition of new
XAI techniques. The core logic is abstracted behind a factory function for
convenient access from the main application pipeline.

Key components:
- `get_explainer`: A factory function to instantiate a specific XAI method.
- `Explainer`: A base class for all explainer implementations, defining a common interface.
"""

from .explainer import Explainer, get_explainer

__all__ = [
    "Explainer",
    "get_explainer",
]
```