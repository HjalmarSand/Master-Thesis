# AI/roll_angle_estimator.py

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# 1. Custom PyTorch Dataset
class RollAngleDataset(Dataset):
    """Custom Dataset for loading segmented fisheye images and roll angles."""

    def __init__(self, csv_file: Path, images_root_dir: Path, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.images_root_dir = images_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        dataset_num = self.annotations.iloc[index, 0]
        image_idx = self.annotations.iloc[index, 1]
        roll_angle = torch.tensor(float(self.annotations.iloc[index, 2]))

        base_filename_cam1 = f"binary_distorted_cam1_{image_idx:04d}.png"
        img_path = (
            self.images_root_dir
            / f"dataset{dataset_num}"
            / "segmented_fisheye_cam1"
            / base_filename_cam1
        )

        if not img_path.exists():
            base_filename_cam2 = f"binary_distorted_cam2_{image_idx:04d}.png"
            img_path = (
                self.images_root_dir
                / f"dataset{dataset_num}"
                / "segmented_fisheye_cam2"
                / base_filename_cam2
            )
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Could not find image for index {image_idx} in dataset {dataset_num}"
                )

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return (image, roll_angle)


# 2. Simple CNN Model for Regression
class RollEstimatorCNN(nn.Module):
    """A simple CNN to predict a single continuous value (roll angle)."""

    # input 224x224
    def __init__(self):
        super(RollEstimatorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> 16x112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32x56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32x28x28
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64x14x14
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# 3. Main Training Function
def train_roll_estimator(
    images_root_dir: Path,
    annotation_csv: Path,
    model_save_path: Path,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    Main function to orchestrate the training of the roll estimation model.
    """
    print("\nStarting Roll Angle Estimator Training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            # Resize shorter edge to 256 pixels, keeping aspect ratio for the 1280x960 fisheye image size
            transforms.Resize(256),
            # Crop a 224x224 square from the center
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize for single channel image
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_dataset = RollAngleDataset(
        csv_file=annotation_csv, images_root_dir=images_root_dir, transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total annotated images: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = RollEstimatorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, angles in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
        ):
            images, angles = images.to(device), angles.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, angles in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]  "
            ):
                images, angles = (
                    images.to(device),
                    angles.to(device).float().unsqueeze(1),
                )
                outputs = model(images)
                loss = criterion(outputs, angles)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining complete. Model saved to '{model_save_path}'")
    return 0
