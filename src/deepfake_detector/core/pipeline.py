# src/deepfake_detector/core/pipeline.py

"""
Defines the main detection pipeline class.

This class orchestrates the full process: loading a model, preprocessing an
input image, running inference, and generating an explanation map using an
XAI module.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import Image
from torchvision import transforms as T

# Assuming a model definition exists in models.model
# This is a placeholder for the actual model architecture import.
# The get_model function should return an initialized model instance.
from ..models import get_model

# --- Constants ---
# These could be moved to a configuration file (e.g., core/config.py)
IMG_SIZE = (224, 224)
# Standard normalization for models pre-trained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Orchestrates the deepfake detection process.

    This class encapsulates the entire pipeline from image preprocessing to
    model inference and explainability map generation. It is designed to be
    initialized once and used for multiple predictions.
    """

    def __init__(self, model_path: Union[str, Path], device: str = "cpu"):
        """
        Initializes the detection pipeline.

        Args:
            model_path (Union[str, Path]): Path to the pre-trained model checkpoint (.pth).
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.model = self._load_model()
        self.explainer = IntegratedGradients(self.model)
        self.transforms = self._get_transforms()

        logger.info(f"DetectionPipeline initialized on device '{self.device}'")

    def _load_model(self) -> torch.nn.Module:
        """Loads the model architecture and state dictionary from the checkpoint file."""
        logger.info(f"Loading model from {self.model_path}...")
        # The get_model function should return the model architecture
        # without pre-trained weights, ready to load a state_dict.
        model = get_model(pretrained=False)

        try:
            # Load the state dictionary, ensuring it's mapped to the correct device
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Handle potential 'module.' prefix if model was saved with DataParallel
            if all(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _get_transforms(self) -> T.Compose:
        """Defines the image preprocessing pipeline."""
        return T.Compose(
            [T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocesses a single image for model input.

        Args:
            image (Image.Image): The input PIL Image.

        Returns:
            torch.Tensor: The processed image tensor, ready for the model.
        """
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transforms(image)
        # Add batch dimension and move to the correct device
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess_explanation(self, attributions: torch.Tensor) -> np.ndarray:
        """
        Converts raw attributions from Captum into a visualizable heatmap.

        Args:
            attributions (torch.Tensor): The raw attribution tensor from the explainer.

        Returns:
            np.ndarray: A normalized grayscale heatmap as a NumPy array (0-255, uint8).
        """
        # Squeeze batch dimension, move to CPU, and convert to numpy
        attr = attributions.squeeze(0).cpu().detach().numpy()
        # Sum attributions across color channels and take absolute values
        attr = np.transpose(attr, (1, 2, 0))
        attr = np.sum(np.abs(attr), axis=-1)
        # Normalize to 0-1 range
        attr_min, attr_max = np.min(attr), np.max(attr)
        if attr_max > 0:
            attr = (attr - attr_min) / (attr_max - attr_min)
        # Scale to 0-255 and convert to uint8
        heatmap = (attr * 255).astype(np.uint8)
        return heatmap

    def predict(
        self, image_source: Union[str, Path, bytes, io.BytesIO]
    ) -> Dict[str, Any]:
        """
        Runs the full detection and explanation pipeline on a single image.

        Args:
            image_source (Union[str, Path, bytes, io.BytesIO]): The source of the image,
                can be a file path or a byte stream.

        Returns:
            Dict[str, Any]: A dictionary containing the prediction results:
                - 'prediction': 'real' or 'fake' (str)
                - 'label': 0 for 'real', 1 for 'fake' (int)
                - 'confidence': The model's confidence in the prediction (float)
                - 'raw_score': The raw sigmoid output score (float)
                - 'explanation_map': A 2D NumPy array representing the heatmap (np.ndarray)
        """
        try:
            image = Image.open(image_source)
        except Exception as e:
            logger.error(f"Could not open image source: {e}")
            raise IOError(f"Could not open image source: {e}")

        # 1. Preprocess image
        input_tensor = self._preprocess(image)
        input_tensor.requires_grad_()  # Required for gradient-based explainers

        # 2. Run model inference
        # We don't use `torch.no_grad()` here because we need gradients for Captum
        output = self.model(input_tensor)

        # Assuming the model returns raw logits for a single class (fake)
        # Apply sigmoid to get a probability-like score
        score = torch.sigmoid(output).item()

        # 3. Generate explanation map
        # We explain the model's output with respect to the "fake" class.
        # For a single-output model, the target index is 0.
        attributions = self.explainer.attribute(input_tensor, target=0)
        explanation_map = self._postprocess_explanation(attributions)

        # 4. Format results
        label = 1 if score >= 0.5 else 0
        prediction = "fake" if label == 1 else "real"
        confidence = score if label == 1 else 1 - score

        result = {
            "prediction": prediction,
            "label": label,
            "confidence": float(f"{confidence:.4f}"),
            "raw_score": float(f"{score:.4f}"),
            "explanation_map": explanation_map,
        }

        logger.info(
            f"Prediction complete: {result['prediction']} (Confidence: {result['confidence']})"
        )
        return result


if __name__ == "__main__":
    # This is an example of how to use the pipeline.
    # It requires a dummy model file and a dummy image to run.
    # To run this script directly for testing, execute:
    # python -m src.deepfake_detector.core.pipeline

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Create dummy files for demonstration ---
    DUMMY_MODEL_PATH = Path("dummy_model.pth")
    DUMMY_IMAGE_PATH = Path("dummy_image.png")
    DUMMY_HEATMAP_PATH = Path("dummy_heatmap.png")

    try:
        # Create a dummy model and save its state_dict
        # In a real scenario, get_model would be a more complex architecture
        dummy_model = get_model(pretrained=False)
        torch.save(dummy_model.state_dict(), DUMMY_MODEL_PATH)

        # Create a dummy image
        dummy_image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image_array, "RGB")
        dummy_image.save(DUMMY_IMAGE_PATH)

        # --- Use the pipeline ---
        logger.info("Initializing pipeline with dummy model...")
        # Use 'cpu' for this example to avoid dependency on CUDA
        pipeline = DetectionPipeline(model_path=DUMMY_MODEL_PATH, device="cpu")

        logger.info(f"Running prediction on dummy image: {DUMMY_IMAGE_PATH}")
        prediction_result = pipeline.predict(DUMMY_IMAGE_PATH)

        print("\n--- Prediction Result ---")
        print(f"Prediction: {prediction_result['prediction']}")
        print(f"Confidence: {prediction_result['confidence']:.2f}")
        print(f"Raw Score: {prediction_result['raw_score']:.2f}")
        print(f"Explanation Map Shape: {prediction_result['explanation_map'].shape}")
        print(f"Explanation Map Type: {prediction_result['explanation_map'].dtype}")

        # To visualize the heatmap (requires matplotlib)
        try:
            import matplotlib.pyplot as plt

            plt.imshow(
                prediction_result["explanation_map"], cmap="hot", interpolation="nearest"
            )
            plt.title("Explanation Heatmap")
            plt.axis("off")
            plt.savefig(DUMMY_HEATMAP_PATH)
            logger.info(f"Saved dummy heatmap to {DUMMY_HEATMAP_PATH}")
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot visualize heatmap.")

    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Error during example run: {e}")
        logger.error(
            "This example requires a model definition in 'src/deepfake_detector/models/model.py'."
        )
        logger.error("Please ensure the model structure is defined to run this example.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Clean up dummy files ---
        if DUMMY_MODEL_PATH.exists():
            DUMMY_MODEL_PATH.unlink()
        if DUMMY_IMAGE_PATH.exists():
            DUMMY_IMAGE_PATH.unlink()
        if DUMMY_HEATMAP_PATH.exists():
            DUMMY_HEATMAP_PATH.unlink()
        logger.info("Cleaned up dummy files.")
```