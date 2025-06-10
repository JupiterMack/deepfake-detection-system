# scripts/train.py
"""
Command-line script to start the model training process.

This script orchestrates the training of the deepfake detection model. It handles:
- Parsing command-line arguments for hyperparameters and paths.
- Setting up the environment (device, random seeds).
- Initializing the data loaders, model, and optimizer.
- Invoking the training loop defined in the Trainer class.

Example Usage:
---------------
From the project root directory:

python -m scripts.train \
    --data_dir /path/to/your/dataset \
    --model_save_path artifacts/models/deepfake_detector_v1.pth \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
import logging

# Add the project root to the Python path to allow for absolute imports
# This is necessary to run the script directly, e.g., `python scripts/train.py`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.deepfake_detector.data_loader import get_dataloaders
    from src.deepfake_detector.model import DeepfakeDetector
    from src.deepfake_detector.training.trainer import Trainer
except ImportError as e:
    print(f"Error: Failed to import project modules. \n"
          f"Please ensure the 'src' directory is in your PYTHONPATH or run the script as a module from the project root, e.g.,\n"
          f"python -m scripts.train [ARGS]\n"
          f"Original Error: {e}")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across different libraries.
    
    Args:
        seed (int): The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Using deterministic algorithms can slow down training but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudunn.benchmark = False
    logging.info(f"Random seed set to {seed} for reproducibility.")

def parse_arguments():
    """
    Parses command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Deepfake Detection Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Dataset and Model Paths ---
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root directory of the dataset.')
    parser.add_argument('--model_save_path', type=str, required=True,
                        help='Path to save the trained model weights (e.g., artifacts/models/model.pth).')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of the dataset to use for validation.')

    # --- Model and System Configuration ---
    parser.add_argument('--model_arch', type=str, default='xception',
                        help='Model architecture to use (e.g., xception, resnet50).')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training even if available.')

    return parser.parse_args()

def main():
    """
    Main function to orchestrate the model training process.
    """
    args = parse_arguments()

    logging.info("--- Deepfake Detection System - Model Training ---")
    logging.info("Configuration:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  - {arg}: {value}")
    
    # --- Environment Setup ---
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info("Initializing data loaders...")
    try:
        train_loader, val_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            seed=args.seed
        )
        logging.info(f"Data loaders created. Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    except FileNotFoundError:
        logging.error(f"Dataset directory not found at: {args.data_dir}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while creating data loaders: {e}", exc_info=True)
        sys.exit(1)

    # --- Model Initialization ---
    logging.info(f"Initializing model with architecture: '{args.model_arch}'")
    try:
        # Assuming the model class can take the architecture name and load a pretrained backbone
        model = DeepfakeDetector(model_arch=args.model_arch, pretrained=True)
        model.to(device)
    except Exception as e:
        logging.error(f"Failed to initialize the model: {e}", exc_info=True)
        sys.exit(1)

    # --- Trainer Initialization and Execution ---
    logging.info("Initializing the trainer...")
    try:
        # The Trainer class will encapsulate the optimizer, loss function, and training loop
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.lr,
            model_save_path=args.model_save_path
        )
        
        logging.info("Starting model training...")
        trainer.train()
        logging.info("Training finished successfully.")
        logging.info(f"Best model and training history saved to directory of: {args.model_save_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the training process: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
```