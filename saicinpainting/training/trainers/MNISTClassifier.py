import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import argparse
import random
import numpy as np
from collections import Counter
from PIL import ImageDraw

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
    
def test_classifier(classifier, num_samples=20, num_trash_samples=10):
    """
    Tests the trained MNIST classifier on both digit (0-9) and trash (10) images.

    Args:
        classifier (UnifiedRGBMNISTClassifier): The trained classifier instance.
        num_samples (int): Number of test digit images to evaluate.
        num_trash_samples (int): Number of test trash images to evaluate.
    """
    # **Transforms for MNIST (0-9)**
    mnist_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # **Transforms for Trash Class (10)**
    trash_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # **Load MNIST Test Data (0-9)**
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

    # **Load Trash Test Data (Class 10)**
    trash_test = generate_trash_samples(num_trash_samples, trash_transform)
    trash_loader = torch.utils.data.DataLoader(trash_test, batch_size=1, shuffle=True)

    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    misclassified_images = []

    print("\n🎯 **Testing MNIST RGB Classifier on Digits & Trash...**")

    # **Test on Digit Images (0-9)**
    for i, (image, label) in enumerate(mnist_loader):
        if i >= num_samples:
            break

        image_pil = transforms.ToPILImage()(image.squeeze(0))
        predicted_label, confidence = classifier.predict(image_pil)

        class_total[label.item()] += 1
        if predicted_label == label.item():
            correct += 1
            class_correct[label.item()] += 1
        else:
            misclassified_images.append((image_pil, label.item(), predicted_label, confidence))

        print(f"Digit {i}: True Label: {label.item()}, Predicted: {predicted_label}, Confidence: {confidence:.4f}")

        total += 1

    # **Test on Trash Class (10)**
    for i, (image, label) in enumerate(trash_loader):
        image_pil = transforms.ToPILImage()(image.squeeze(0))
        predicted_label, confidence = classifier.predict(image_pil)

        class_total[10] += 1
        if predicted_label == 10:
            correct += 1
            class_correct[10] += 1
        else:
            misclassified_images.append((image_pil, 10, predicted_label, confidence))

        print(f"Trash {i}: True Label: 10, Predicted: {predicted_label}, Confidence: {confidence:.4f}")

        total += 1

    # **Compute & Print Final Accuracy**
    accuracy = 100.0 * correct / total
    print(f"\n✅ **Final Accuracy on {total} samples: {accuracy:.2f}%**")

    # **Per-Class Accuracy**
    print("\n📊 **Per-Class Accuracy**")
    for class_label in range(11):
        if class_total[class_label] > 0:
            class_acc = 100.0 * class_correct[class_label] / class_total[class_label]
            print(f"Class {class_label}: {class_acc:.2f}% ({class_correct[class_label]}/{class_total[class_label]})")

    # # **Show Misclassified Samples**
    # if misclassified_images:
    #     print("\n❌ **Misclassified Samples**")
    #     show_misclassified_images(misclassified_images)


def generate_trash_samples(num_samples, transform):
    """Generate synthetic trash images (random noise, blank images, marks)."""
    trash_images = []
    for _ in range(num_samples):
        img_type = random.choice(["noise", "blank", "marks"])
        img = Image.new("RGB", (28, 28), (0, 0, 0))  # Default blank image

        if img_type == "noise":
            img = Image.fromarray(np.uint8(np.random.rand(28, 28, 3) * 255))
        elif img_type == "marks":
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(1, 5)):
                x1, y1, x2, y2 = random.randint(0, 27), random.randint(0, 27), random.randint(0, 27), random.randint(0, 27)
                draw.line((x1, y1, x2, y2), fill=(255, 255, 255), width=1)

        img = transform(img)
        trash_images.append((img, 10))  # Assign label 10 (trash)

    return trash_images


def main():
    parser = argparse.ArgumentParser(description="Test RGB MNIST Classifier")
    parser.add_argument("--model_path", type=str, default="mnist_equation/mnist_classifier/models/mnist_classifier.pt",
                        help="Path to trained classifier model")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test images to evaluate")
    args = parser.parse_args()


    # Load trained classifier
    classifier = UnifiedRGBMNISTClassifier(model_path=args.model_path)

    # Run test
    test_classifier(classifier, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
    # from collections import Counter
    # import pandas as pd

    # train_labels_file = "data/multi_digit_classifier/test/test_labels.csv"
    # df = pd.read_csv(train_labels_file, header=None, names=["image", "label"])
    # print("Class distribution in test data:", Counter(df["label"]))