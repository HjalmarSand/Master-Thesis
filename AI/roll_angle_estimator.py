# skyline_processing/roll_angle_estimator.py

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RollAngleDataset(Dataset):
    """Custom Dataset for loading segmented fisheye images and roll angles."""

    def __init__(self, csv_file: Path, image_dir: Path, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = self.image_dir / img_name

        # Load image as grayscale
        image = Image.open(img_path).convert("L")

        # Get roll angle and convert to a tensor
        roll_angle = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, roll_angle)


class RollEstimatorCNN(nn.Module):
    """A simple CNN to predict a single continuous value (roll angle)."""

    def __init__(self):
        super(RollEstimatorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=5, stride=2, padding=2
            ),  # Input is 1 channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256),  # Input size depends on final conv output
            nn.ReLU(),
            nn.Linear(256, 1),  # Output a single value for the roll angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


def train_roll_estimator(
    image_dir: Path,
    annotation_csv: Path,
    model_save_path: Path,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    Main function to orchestrate the training of the roll estimation model.
    """
    print("\n--- Starting Roll Angle Estimator Training ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define transforms for the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for single channel
        ]
    )

    # Create dataset and dataloader
    dataset = RollAngleDataset(
        csv_file=annotation_csv, image_dir=image_dir, transform=transform
    )
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = RollEstimatorCNN().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error is good for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, angles in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            angles = (
                angles.to(device).float().unsqueeze(1)
            )  # Reshape for loss calculation

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, angles)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete. Model saved to '{model_save_path}'")
    return 0
