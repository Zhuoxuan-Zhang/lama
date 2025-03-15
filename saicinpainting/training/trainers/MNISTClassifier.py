import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class RGBMNISTClassifier(nn.Module):
    """CNN-based classifier for RGB MNIST images (digits 0-9 + 'trash' class, 11 total)."""

    def __init__(self, num_classes=11):
        super(RGBMNISTClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Adjusted for 3 input channels
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 3x3 feature map
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # 11 output classes (0-9 + trash)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x  # Logits (no softmax for stability)


class UnifiedRGBMNISTClassifier:
    """
    A classifier that predicts if an image is a MNIST digit (0-9) 
    or belongs to the 'trash' class (non-digit) using RGB images.
    """

    def __init__(self, model_path="/users/zzhan513/data/zzhan513/visual_reasoning/BENCHMARKING-VISUAL-GENERATIVE-REASONING/mnist_equation/mnist_classifier/models/mnist_classifier.pt"):
        """
        Initializes the classifier by loading a single CNN model.

        Args:
            model_path (str): Path to the classifier checkpoint.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.num_classes = 11  # 10 digits + 1 trash class
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Keep RGB format
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize for RGB input
        ])

    def _load_model(self):
        """
        Loads the trained classifier model.
        """
        model = RGBMNISTClassifier(num_classes=self.num_classes)
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Classifier model not found at {self.model_path}")

    def predict(self, image):
        """
        Predicts the class of the given image.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            int: Predicted digit (0-9) or 10 for 'trash' (non-digit).
            float: Confidence score.
        """
        # Apply transformations
        if isinstance(image, Image.Image):
            img = self.transform(image)
        elif isinstance(image, torch.Tensor):
            img = image
        else:
            raise TypeError("Input must be a PIL image or a torch Tensor.")

        # Add batch dimension
        img = img.to(self.device).unsqueeze(0)

        # Predict class probabilities
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            max_confidence, predicted_class = torch.max(probs, dim=-1)

        return predicted_class.item(), max_confidence.item()