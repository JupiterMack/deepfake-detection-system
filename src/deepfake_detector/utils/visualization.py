# src/deepfake_detector/utils/visualization.py

"""
Visualization utilities for the deepfake detection system.

This module provides functions to help visualize model outputs, such as
overlaying heatmaps from explainable AI (XAI) methods onto original images.
This is crucial for understanding which parts of an image the model
considers to be manipulated.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlays a heatmap onto the original image with a specified transparency.

    This function is used to visualize attribution maps from XAI methods.

    Args:
        original_image (np.ndarray): The original input image as a NumPy array.
                                     Expected to be in BGR format (OpenCV default).
                                     Shape: (H, W, 3).
        heatmap (np.ndarray): A 2D NumPy array representing the heatmap/attribution
                              map. Values should be floats. Shape: (H, W).
        colormap (int): The OpenCV colormap to apply to the heatmap.
                        Defaults to cv2.COLORMAP_JET.
        alpha (float): The blending factor for the heatmap. A value of 0.0 means
                       the heatmap is fully transparent, while 1.0 means it's
                       fully opaque. Defaults to 0.5.

    Returns:
        np.ndarray: The original image with the heatmap overlaid, in BGR format.
    """
    if original_image.shape[:2] != heatmap.shape[:2]:
        logger.warning(
            f"Heatmap shape {heatmap.shape} differs from image shape "
            f"{original_image.shape[:2]}. Resizing heatmap."
        )
        heatmap = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )

    # Normalize the heatmap to the 0-255 range
    if np.max(heatmap) > 0:
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = (heatmap * 255).astype(np.uint8)

    # Apply the colormap to the heatmap
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)

    # Blend the heatmap with the original image
    # The formula is: output = (1-alpha)*img1 + alpha*img2
    overlayed_image = cv2.addWeighted(
        colored_heatmap, alpha, original_image, 1 - alpha, 0
    )

    return overlayed_image


def create_explanation_image(
    original_pil_image: Image.Image,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
) -> Image.Image:
    """
    Creates a side-by-side comparison image showing the original image and
    the image with an XAI heatmap overlay.

    Args:
        original_pil_image (Image.Image): The original image in PIL format.
        heatmap (np.ndarray): The 2D attribution map from an XAI method.
        prediction (str): The predicted label (e.g., "FAKE" or "REAL").
        confidence (float): The model's confidence score for the prediction.

    Returns:
        Image.Image: A PIL image containing the side-by-side comparison.
    """
    # Convert PIL image (RGB) to OpenCV format (BGR)
    original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)

    # Generate the overlay image
    overlayed_image = overlay_heatmap(original_cv_image, heatmap, alpha=0.6)

    # Prepare text for annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White
    line_type = 2
    text_bg_color = (0, 0, 0)  # Black

    pred_text = f"Prediction: {prediction.upper()}"
    conf_text = f"Confidence: {confidence:.2%}"

    # Add text with background to the original image
    (w, h), _ = cv2.getTextSize(pred_text, font, font_scale, line_type)
    cv2.rectangle(original_cv_image, (5, 5), (15 + w, 20 + h), text_bg_color, -1)
    cv2.putText(
        original_cv_image,
        pred_text,
        (10, 20 + h // 2),
        font,
        font_scale,
        font_color,
        line_type,
    )
    (w, h), _ = cv2.getTextSize(conf_text, font, font_scale, line_type)
    cv2.rectangle(original_cv_image, (5, 30 + h), (15 + w, 45 + 2 * h), text_bg_color, -1)
    cv2.putText(
        original_cv_image,
        conf_text,
        (10, 35 + 2 * h),
        font,
        font_scale,
        font_color,
        line_type,
    )

    # Add text with background to the overlay image
    title_text = "XAI Explanation"
    (w, h), _ = cv2.getTextSize(title_text, font, font_scale, line_type)
    cv2.rectangle(overlayed_image, (5, 5), (15 + w, 20 + h), text_bg_color, -1)
    cv2.putText(
        overlayed_image,
        title_text,
        (10, 20 + h // 2),
        font,
        font_scale,
        font_color,
        line_type,
    )

    # Combine the two images side-by-side
    combined_image = np.hstack((original_cv_image, overlayed_image))

    # Convert back to PIL format (RGB) for returning
    final_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_image_rgb)
```