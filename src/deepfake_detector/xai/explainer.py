# src/deepfake_detector/xai/explainer.py

"""
Implements the explainability logic for the deepfake detection system.

This module provides a class that wraps around an attribution algorithm from the
Captum library to generate heatmaps (saliency maps). These maps highlight which
pixels in an image were most influential for the model's prediction, providing
insight into the decision-making process.
"""

import logging
from typing import Optional

import numpy as np
import torch
from captum.attr import GradCAM, LayerAttribution
from torch.nn import Module

logger = logging.getLogger(__name__)


class Explainer:
    """
    A wrapper class for generating explanations for model predictions using Captum.

    This class uses Grad-CAM (Gradient-weighted Class Activation Mapping) to
    highlight the regions of an image that were most influential in a model's
    decision. It is initialized with a trained model and can be used to generate
    attribution maps for multiple inputs.
    """

    def __init__(self, model: Module):
        """
        Initializes the Explainer with a trained PyTorch model.

        The constructor identifies the last convolutional layer in the model's
        feature extractor, which is used as the target layer for Grad-CAM.

        Args:
            model (Module): The trained deepfake detection model. The model must
                            have a 'features' attribute which is a Sequential
                            module containing the convolutional layers.

        Raises:
            ValueError: If the model does not have a 'features' attribute or if
                        no convolutional layer can be found within it.
        """
        if not hasattr(model, 'features') or not isinstance(model.features, torch.nn.Sequential):
            raise ValueError("Model must have a 'features' attribute of type nn.Sequential.")

        self.model = model
        self.model.eval()  # Ensure the model is in evaluation mode

        # Automatically find the last convolutional layer for Grad-CAM
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise ValueError("Could not find a Conv2d layer in model.features for Grad-CAM.")

        logger.info(f"Using target layer for Grad-CAM: {target_layer}")
        self.grad_cam = GradCAM(self.model, target_layer)

    def _find_target_layer(self) -> Optional[Module]:
        """Iterates backwards through model features to find the last Conv2d layer."""
        for layer in reversed(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        return None

    def generate_attributions(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generates an attribution map (heatmap) for a given input tensor using Grad-CAM.

        Args:
            input_tensor (torch.Tensor): The preprocessed input image tensor.
                                         It should be a 4D tensor with the
                                         shape (1, C, H, W).
            target_class (int): The target class index for which to generate
                                the explanation. For binary classification,
                                this is typically 0 for 'real' and 1 for 'fake'.
                                Defaults to 1 ('fake').

        Returns:
            np.ndarray: A 2D numpy array representing the heatmap, with values
                        normalized to the range [0, 1]. The shape is (H, W),
                        matching the input tensor's spatial dimensions.

        Raises:
            ValueError: If the input_tensor does not have the expected 4D shape.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(f"Input tensor must have shape (1, C, H, W), but got {input_tensor.shape}")

        # Ensure tensor is on the same device as the model
        try:
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)
        except StopIteration:
            # Handle models with no parameters, though unlikely for CNNs
            logger.warning("Model has no parameters. Assuming CPU device.")
            device = torch.device("cpu")
            input_tensor = input_tensor.to(device)


        # Generate attributions using Grad-CAM.
        # The output from Grad-CAM has the shape of the target layer's feature map.
        attributions = self.grad_cam.attribute(input_tensor, target=target_class)

        # Upsample the attributions to the original image size for overlaying.
        # Captum's interpolate helper makes this straightforward.
        upsampled_attributions = LayerAttribution.interpolate(
            attributions, input_tensor.shape[2:], interpolate_mode='bilinear'
        )

        # Process the attributions for visualization:
        # 1. Squeeze to remove batch and channel dimensions -> (H, W)
        # 2. Move to CPU and convert to a NumPy array.
        # 3. Apply ReLU to keep only positive influences, which are what Grad-CAM
        #    is designed to show.
        # 4. Normalize the heatmap to [0, 1] for consistent visualization.
        heatmap = upsampled_attributions.squeeze(0).squeeze(0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU

        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max  # Normalize to [0, 1]

        logger.info("Successfully generated Grad-CAM attribution map.")
        return heatmap

```