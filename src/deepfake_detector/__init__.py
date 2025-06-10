"""
Deepfake Detector Package.

This package provides the core functionalities for the deepfake detection system,
including data processing, model definitions, training, inference, and explainability
modules.

The main components are exposed at the top level for easy access.
"""

import logging
import os
import sys

# --- Package Metadata ---
__version__ = "0.1.0"
__author__ = "The Deepfake Detection System Development Team"
__email__ = "project-contact@example.com"


# --- Logging Setup ---
# Configure a logger for the entire package.
# This setup allows other modules in the package to use logging without
# needing to configure it themselves. They can just call `logging.getLogger(__name__)`.
# The log level can be controlled by an environment variable.

LOG_LEVEL = os.environ.get("DEEPFAKE_DETECTOR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
logger.info(f"Deepfake Detector package version {__version__} initialized.")
logger.info(f"Log level set to {LOG_LEVEL}.")


# --- Public API Imports ---
# Import key classes and functions from submodules to make them directly
# accessible from the `deepfake_detector` namespace. This simplifies usage for
# external scripts and applications.
#
# We use try-except blocks to prevent ImportErrors during development if a
# submodule has not yet been created. This allows the package to be imported
# even if some components are missing.

try:
    from .pipeline import DetectionPipeline
except ImportError:
    logger.warning(
        "Could not import 'DetectionPipeline'. "
        "Ensure 'src/deepfake_detector/pipeline.py' exists and is correctly implemented."
    )
    DetectionPipeline = None  # type: ignore

try:
    from .models import get_model
except ImportError:
    logger.warning(
        "Could not import 'get_model'. "
        "Ensure 'src/deepfake_detector/models/__init__.py' exists and is correctly implemented."
    )
    get_model = None  # type: ignore

try:
    from .data.preprocessing import preprocess_image
except ImportError:
    logger.warning(
        "Could not import 'preprocess_image'. "
        "Ensure 'src/deepfake_detector/data/preprocessing.py' exists and is correctly implemented."
    )
    preprocess_image = None  # type: ignore

try:
    from .explainability import generate_explanation
except ImportError:
    logger.warning(
        "Could not import 'generate_explanation'. "
        "Ensure 'src/deepfake_detector/explainability/__init__.py' exists and is correctly implemented."
    )
    generate_explanation = None  # type: ignore


# --- Define Public API ---
# The `__all__` list explicitly declares the names that should be imported when
# a client uses `from deepfake_detector import *`. It's a good practice for
# package hygiene and helps static analysis tools.

__all__ = [
    # Core components
    "DetectionPipeline",
    "get_model",
    "preprocess_image",
    "generate_explanation",
    # Package utilities
    "logger",
    "__version__",
]

logger.debug("Public API for 'deepfake_detector' package configured.")