# src/deepfake_detector/training/trainer.py

"""
Implements the model training loop, handling epochs, batches, forward/backward
passes, optimization, and logging of training/validation metrics.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the model training and validation process.

    This class encapsulates the entire training loop, including epoch iteration,
    batch processing, forward/backward passes, optimization, metric calculation,
    and model checkpointing. It also supports optional adversarial training using
    the Fast Gradient Sign Method (FGSM).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int,
        checkpoint_dir: Path,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        adversarial_training: bool = False,
        adv_epsilon: float = 0.01,
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The neural network model to train.
            optimizer (optim.Optimizer): The optimizer for updating model weights.
            criterion (nn.Module): The loss function.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): The device to run training on (e.g., 'cuda' or 'cpu').
            num_epochs (int): The total number of epochs to train for.
            checkpoint_dir (Path): Directory to save model checkpoints.
            scheduler (Optional): Learning rate scheduler. Defaults to None.
            adversarial_training (bool): Whether to use adversarial training. Defaults to False.
            adv_epsilon (float): Epsilon value for FGSM adversarial attack. Defaults to 0.01.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.adversarial_training = adversarial_training
        self.adv_epsilon = adv_epsilon

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, list]:
        """
        Starts the main training loop for the specified number of epochs.

        Returns:
            Dict[str, list]: A dictionary containing the history of training and
                             validation metrics (loss and accuracy).
        """
        logger.info("🚀 Starting model training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Adversarial training: {'Enabled' if self.adversarial_training else 'Disabled'}")

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            # Train for one epoch
            train_loss, train_acc = self._train_epoch(epoch)

            # Validate the model
            val_loss, val_acc = self._validate_epoch()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Step the scheduler if one is provided
            if self.scheduler:
                # Special case for ReduceLROnPlateau which needs a metric
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            end_time = time.time()
            epoch_duration = end_time - start_time

            logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Time: {epoch_duration:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)
        
        # Save the final model
        self._save_checkpoint(self.num_epochs, is_best=False, filename="final_model.pth")
        
        logger.info("✅ Training finished.")
        logger.info(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")
        return self.history

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Performs a single training epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Tuple[float, float]: The average training loss and accuracy for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs} [Training]",
            leave=False,
        )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)
            
            if self.adversarial_training:
                # Enable gradients for the input for adversarial attack
                inputs.requires_grad = True

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Standard training step
            loss.backward()

            if self.adversarial_training:
                # Collect gradients of the input
                input_grad = inputs.grad.data
                # Create adversarial example using FGSM
                perturbed_inputs = self._fgsm_attack(inputs, input_grad)
                # Re-classify the perturbed image
                adv_outputs = self.model(perturbed_inputs)
                adv_loss = self.criterion(adv_outputs, labels)
                # Combine losses and backpropagate
                total_adv_loss = loss + adv_loss
                # We need to zero gradients again before the second backward pass
                self.optimizer.zero_grad()
                total_adv_loss.backward()
                running_loss += total_adv_loss.item() * inputs.size(0)
            else:
                running_loss += loss.item() * inputs.size(0)

            self.optimizer.step()

            # Calculate accuracy
            preds = torch.sigmoid(outputs) >= 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{correct_predictions / total_samples:.4f}",
            )

        avg_loss = running_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Performs a single validation epoch.

        Returns:
            Tuple[float, float]: The average validation loss and accuracy for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            self.val_loader, desc="[Validating]", leave=False
        )

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                preds = torch.sigmoid(outputs) >= 0.5
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                progress_bar.set_postfix(
                    loss=f"{running_loss / total_samples:.4f}",
                    acc=f"{correct_predictions / total_samples:.4f}",
                )

        avg_loss = running_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def _fgsm_attack(self, image: torch.Tensor, data_grad: torch.Tensor) -> torch.Tensor:
        """
        Generates an adversarial example using the Fast Gradient Sign Method (FGSM).

        Args:
            image (torch.Tensor): The original input image.
            data_grad (torch.Tensor): The gradient of the loss with respect to the input image.

        Returns:
            torch.Tensor: The perturbed (adversarial) image.
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + self.adv_epsilon * sign_data_grad
        # Clip the perturbed image to maintain the original data range (e.g., [0, 1])
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def _save_checkpoint(self, epoch: int, is_best: bool, filename: str = "best_model.pth"):
        """
        Saves a model checkpoint.

        Args:
            epoch (int): The current epoch number.
            is_best (bool): True if this is the best model so far, False otherwise.
            filename (str): The filename for the checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            filepath = self.checkpoint_dir / filename
            torch.save(state, filepath)
            logger.info(f"💾 Checkpoint saved: {filepath} (Epoch {epoch}, Val Acc: {self.best_val_acc:.4f})")
        elif filename: # For saving final model
            filepath = self.checkpoint_dir / filename
            torch.save(state, filepath)
            logger.info(f"💾 Final model saved: {filepath}")
```