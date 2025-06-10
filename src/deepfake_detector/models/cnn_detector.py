# src/deepfake_detector/models/cnn_detector.py

"""
Defines the deepfake detection model architecture using PyTorch.

This module implements a custom Convolutional Neural Network (CNN) based on
VGG-style principles, tailored for the binary classification task of
distinguishing real images from deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDetector(nn.Module):
    """
    A VGG-style Convolutional Neural Network for deepfake detection.

    The architecture consists of several convolutional blocks followed by a
    fully connected classifier. Each convolutional block contains Conv2d,
    BatchNorm2d, ReLU activation, and MaxPool2d layers. Dropout is used
    in the classifier to prevent overfitting.

    Args:
        input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
    """

    def __init__(self, input_channels: int = 3):
        super(CNNDetector, self).__init__()

        # Feature extractor: Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            # Input: (batch_size, 3, 224, 224)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, 112, 112)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 64, 56, 56)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 128, 28, 28)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 256, 14, 14)

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 512, 7, 7)
        )

        # Adaptive pooling to handle different input sizes gracefully,
        # ensuring the output feature map size is always 7x7.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier: Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1),  # Output is a single logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, 1).
                          A sigmoid function should be applied to these logits to get
                          probabilities.
        """
        x = self.features(x)
        x = self.avgpool(x)
        # Flatten the feature map
        # The view call reshapes the tensor. -1 infers the batch size.
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # This block is for demonstration and basic testing of the model architecture.
    # It will only run when the script is executed directly.

    print("--- Testing CNNDetector Model Architecture ---")

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model and move it to the selected device
    model = CNNDetector().to(device)
    print("\nModel Architecture:")
    print(model)

    # Create a dummy input tensor to simulate a batch of images
    # Shape: (batch_size, channels, height, width)
    # Common input size for image models is 224x224
    try:
        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        print(f"\nShape of dummy input tensor: {dummy_input.shape}")

        # Perform a forward pass
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(dummy_input)

        print(f"Shape of model output (logits): {output.shape}")

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(output)
        print(f"Shape of probabilities: {probabilities.shape}")
        print(f"Example output probabilities:\n{probabilities.cpu().numpy()}")

        # Calculate the number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("\n--- Model Test Successful ---")

    except Exception as e:
        print(f"\n--- An error occurred during model testing: {e} ---")