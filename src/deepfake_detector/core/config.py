# src/deepfake_detector/core/config.py

"""
Centralized configuration management for the Deepfake Detection System.

This module loads environment variables from a .env file and provides a
Settings object with sensible defaults for all configuration parameters.
This ensures that all parts of the application use a consistent configuration.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- Base Directory ---
# Define the project's base directory.
# The config.py file is at src/deepfake_detector/core/config.py,
# so the project root is three levels up.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# --- Load Environment Variables ---
# Load environment variables from a .env file at the project root.
# This allows for easy configuration for development and production environments
# without modifying the source code. The .env file is ignored by Git.
load_dotenv(BASE_DIR / ".env")


class Settings:
    """
    A class to hold all application settings, read from environment variables.

    Attributes are defined at the class level and can be accessed through the
    globally available `settings` instance.
    e.g., `from deepfake_detector.core.config import settings`
          `print(settings.PROJECT_NAME)`
    """
    # --- Project Metadata ---
    PROJECT_NAME: str = "Deepfake Detection System"
    VERSION: str = "0.1.0"

    # --- Environment and Paths ---
    # Note: Paths are constructed using the BASE_DIR for robustness.
    # The corresponding directories should be created by the components that use them.
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    MODELS_DIR: Path = BASE_DIR / os.getenv("MODELS_DIR", "models")
    OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
    LOGS_DIR: Path = BASE_DIR / os.getenv("LOGS_DIR", "logs")

    # Path to the specific model file used for inference.
    # This can be overridden by setting the MODEL_PATH environment variable.
    # The default path is constructed relative to the MODELS_DIR.
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "deepfake_model_final.pth")))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Model and Training Parameters ---
    # Determine the device to use for training/inference (CUDA or CPU).
    # This automatically selects the GPU if available.
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Image preprocessing settings.
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", 224))

    # Dataloader settings.
    # Defaults to the number of available CPU cores if not set.
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", os.cpu_count() or 1))

    # Training hyperparameters. These can be overridden via environment variables
    # or command-line arguments in the training script.
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 1e-4))
    EPOCHS: int = int(os.getenv("EPOCHS", 25))

    # Reproducibility.
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", 42))


# --- Global Settings Instance ---
# Create a single, globally accessible instance of the Settings class.
# Other modules should import this instance to access configuration values.
settings = Settings()

# Example of how to use the settings object.
# This block will only run if the script is executed directly.
if __name__ == '__main__':
    print("--- Deepfake Detector Configuration ---")
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Base Directory: {settings.BASE_DIR}")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Device: {settings.DEVICE}")
    print(f"Image Size: {settings.IMAGE_SIZE}")
    print(f"Batch Size: {settings.BATCH_SIZE}")
    print(f"Learning Rate: {settings.LEARNING_RATE}")
    print("---------------------------------------")

```