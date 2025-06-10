# scripts/predict.py
"""
Command-line script for running inference on a single image to detect deepfakes.

This script loads a pre-trained deepfake detection model, processes a given
input image, and outputs a prediction (real or fake) along with a confidence
score. It can also generate and save an explanation heatmap using Captum,
highlighting the image regions that most influenced the model's decision.

Usage:
    python scripts/predict.py \
        --image_path /path/to/your/image.jpg \
        --model_path /path/to/your/model.pth \
        --output_path /path/to/save/heatmap.jpg

Arguments:
    --image_path: Path to the input image file.
    --model_path: Path to the trained PyTorch model checkpoint (.pth file).
    --output_path (optional): Path to save the generated explanation heatmap.
    --device (optional): Device to run inference on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
"""

import argparse
import sys
import os
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from captum.attr import IntegratedGradients

# Add project root to Python path to allow importing from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# In a real project, the model definition would be in a separate file, e.g., 'src/model.py'
# To make this script self-contained, a placeholder is defined here.
# from src.model import DeepfakeDetector # Ideal import
class DeepfakeDetector(nn.Module):
    """
    A placeholder model definition.
    The actual model architecture used for training should be defined here or imported.
    This example uses a simple CNN architecture.
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Example architecture (e.g., based on ResNet or a custom CNN)
        # This should match the architecture of the saved model_path
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Assuming input image size of 224x224, after 3 max-pools, size is 28x28
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads a pre-trained model from a .pth file.

    Args:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        nn.Module: The loaded model.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = DeepfakeDetector()
        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def preprocess_image(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """
    Loads and preprocesses an image for model inference.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Tuple[torch.Tensor, Image.Image]: A tuple containing the preprocessed
                                           image tensor and the original PIL image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    # Define the same transformations used during training
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = preprocess_transform(image)
    # Add a batch dimension (B, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor, image


def generate_and_save_explanation(model: nn.Module, input_tensor: torch.Tensor, original_image: Image.Image, output_path: str, device: torch.device):
    """
    Generates an explanation heatmap using Integrated Gradients and saves it.

    Args:
        model (nn.Module): The loaded model.
        input_tensor (torch.Tensor): The preprocessed input tensor for the model.
        original_image (Image.Image): The original image (for overlay).
        output_path (str): The path to save the heatmap image.
        device (torch.device): The device the model is on.
    """
    print("Generating explanation heatmap...")
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    # Initialize the attribution algorithm
    ig = IntegratedGradients(model)

    # Calculate attributions. Target=0 for 'real', 1 for 'fake'.
    # We want to see what makes the model think the image is 'fake'.
    attributions, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attributions = attributions.squeeze(0).cpu().detach().numpy()

    # Process attributions for visualization
    # Sum across color channels and take absolute values
    attributions = np.transpose(attributions, (1, 2, 0))
    heatmap = np.abs(attributions).sum(axis=2)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) # Normalize to [0, 1]

    # Convert original image to OpenCV format
    original_cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    original_cv_image = cv2.resize(original_cv_image, (224, 224))

    # Apply colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap on the original image
    alpha = 0.5
    overlaid_image = cv2.addWeighted(heatmap_colored, alpha, original_cv_image, 1 - alpha, 0)

    try:
        cv2.imwrite(output_path, overlaid_image)
        print(f"Explanation heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap image: {e}", file=sys.stderr)


def main():
    """Main function to run the prediction script."""
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference Script")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--output_path', type=str, help="Optional. Path to save the explanation heatmap image.")
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help="Device to use ('cpu' or 'cuda'). Defaults to 'cuda' if available.")

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = load_model(args.model_path, device)

    # Preprocess the image
    input_tensor, original_image = preprocess_image(args.image_path)
    input_tensor = input_tensor.to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        # Apply sigmoid to get probability. Assumes binary classification with sigmoid output.
        probability = torch.sigmoid(output).item()

    # Determine prediction and confidence
    # Assuming the model outputs logits where > 0 corresponds to 'fake'
    if probability > 0.5:
        prediction = "Fake"
        confidence = probability
    else:
        prediction = "Real"
        confidence = 1 - probability

    # Print results
    print("\n--- Prediction Results ---")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print("--------------------------\n")

    # Generate and save explanation if output path is provided
    if args.output_path:
        generate_and_save_explanation(model, input_tensor, original_image, args.output_path, device)


if __name__ == '__main__':
    main()
```