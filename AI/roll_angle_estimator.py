import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from tqdm import tqdm


# 1. Custom PyTorch Dataset
class RollAngleDataset(Dataset):
    """
    A robust Dataset that scans a LIST of image directories and maps
    filenames to roll angles from an annotation file.
    """

    def __init__(self, image_dirs: List[Path], annotations_csv: Path, transform=None):
        self.transform = transform

        # 1. Load annotations and create an efficient lookup dictionary
        annotations_df = pd.read_csv(annotations_csv)
        self.angle_lookup = {
            (row.dataset, row.index): row.angle for row in annotations_df.itertuples()
        }

        # 2. Scan all provided directories and create a unified list of data points
        self.data_points = []
        for image_dir in image_dirs:
            # Extract the dataset number from the parent folder name (e.g., 'dataset1' -> 1)
            try:
                dataset_num_str = re.search(r"(\d+)", image_dir.parent.name).group(1)
                dataset_num = int(dataset_num_str)
            except (AttributeError, ValueError):
                print(
                    f"Warning: Could not extract dataset number from path: {image_dir.parent.name}. Skipping folder."
                )
                continue

            # Find all images in the current directory
            extensions = ["*.png", "*.jpg", "*.jpeg"]
            image_paths = []
            for ext in extensions:
                image_paths.extend(image_dir.glob(ext))

            # Match each image with its annotation
            for path in sorted(image_paths):
                match = re.search(r"(\d{4})", path.name)
                if match:
                    image_idx = int(match.group(1))
                    lookup_key = (dataset_num, image_idx)

                    if lookup_key in self.angle_lookup:
                        angle = self.angle_lookup[lookup_key]
                        self.data_points.append((path, angle))

        if not self.data_points:
            raise ValueError(
                "No images in the provided directories matched any entries in the annotation file."
            )

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        # The rest of the function logic is the same
        img_path, roll_angle = self.data_points[index]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        _, height, width = image.shape
        center_x, center_y = width // 2, height // 2
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        scale = width / 2.5
        dist_from_center = torch.sqrt(
            ((X - center_x) ** 2 + (Y - center_y) ** 2).float()
        )
        focus_mask = torch.exp(-(dist_from_center**2) / (2 * scale**2))
        image = image * focus_mask.unsqueeze(0)

        return (image, torch.tensor(float(roll_angle)))


# 2. Simple CNN Model for Regression
class RollEstimatorCNN(nn.Module):
    def __init__(self):
        super(RollEstimatorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> 16x112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x56x56
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32x56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32x28x28
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64x14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> 128x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 128x7x7
        )
        self.flatten = nn.Flatten()
        # --- UPDATE THE INPUT SIZE FOR THE LINEAR LAYER ---
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),  # Was 64 * 14 * 14
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# 3. Main Training Function
def train_roll_estimator(
    image_dirs: List[Path],  # Takes a list of directories
    annotation_csv: Path,
    model_save_path: Path,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    print("\nStarting Roll Angle Estimator Training")
    print(f"Using data from {len(image_dirs)} source folder(s).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # The Dataset class now handles the combined data from all directories
    full_dataset = RollAngleDataset(
        annotations_csv=annotation_csv, image_dirs=image_dirs, transform=transform
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Found {len(full_dataset)} annotated images in the directory.")
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


def visualize_model_features(model_path: Path, image_path: Path, output_dir: Path):
    """
    Loads a trained model and visualizes its first-layer filters and the
    feature maps for a given sample image.
    """
    print("\n--- Visualizing Model Features ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the trained model
    if not model_path.exists():
        print(f"Model file not found at '{model_path}'. Please train the model first.")
        return

    model = RollEstimatorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained model from '{model_path.name}'")

    # 2. Visualize the first layer's filters (weights)
    filters = model.conv_layers[0].weight.detach().cpu()
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("First Layer Convolutional Filters", fontsize=16)
    for i, ax in enumerate(axs.flat):
        if i < len(filters):
            ax.imshow(filters[i, 0], cmap="gray")
            ax.set_title(f"Filter {i + 1}")
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    filter_save_path = output_dir / "filters_visualization.png"
    plt.savefig(filter_save_path)
    plt.close()
    print(f"✅ Saved filter visualization to '{filter_save_path}'")

    # 3. Visualize the feature maps for a sample image
    if not image_path.exists():
        print(
            f"Sample image not found at '{image_path}'. Cannot visualize feature maps."
        )
        return

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    # Get the output of the first convolutional layer + ReLU
    feature_maps = model.conv_layers[0](tensor)
    feature_maps = model.conv_layers[1](feature_maps)
    feature_maps = feature_maps.detach().cpu()

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"Feature Maps for {image_path.name}", fontsize=16)
    for i, ax in enumerate(axs.flat):
        if i < feature_maps.shape[1]:
            ax.imshow(feature_maps[0, i], cmap="gray")
            ax.set_title(f"Feature Map {i + 1}")
        ax.axis("off")

    feature_map_save_path = output_dir / "feature_maps_visualization.png"
    plt.savefig(feature_map_save_path)
    plt.close()
    print(f"✅ Saved feature map visualization to '{feature_map_save_path}'")


# In AI/roll_angle_estimator.py


def run_kfold_cross_validation(
    image_dirs: List[Path],
    annotation_csv: Path,
    k_folds: int = 5,
    num_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    Performs K-Fold Cross-Validation and prints the final performance metrics.
    """
    print(f"\n--- Starting {k_folds}-Fold Cross-Validation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_dataset = RollAngleDataset(
        annotations_csv=annotation_csv, image_dirs=image_dirs, transform=transform
    )

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print("\n" + "=" * 50)
        print(f"Fold {fold + 1} / {k_folds}")
        print("=" * 50)

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(
            full_dataset, batch_size=batch_size, sampler=val_sampler
        )

        model = RollEstimatorCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            for images, angles in train_loader:
                images, angles = (
                    images.to(device),
                    angles.to(device).float().unsqueeze(1),
                )
                outputs = model(images)
                loss = criterion(outputs, angles)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                images, angles = (
                    images.to(device),
                    angles.to(device).float().unsqueeze(1),
                )
                outputs = model(images)
                loss = criterion(outputs, angles)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"-> Fold {fold + 1} Validation Loss: {avg_val_loss:.4f}")
        fold_results[fold] = avg_val_loss

    print("\n" + "=" * 50)
    print("Cross-Validation Final Results")
    print("=" * 50)
    avg_loss = np.mean(list(fold_results.values()))
    std_loss = np.std(list(fold_results.values()))
    print(f"Average Validation Loss across {k_folds} folds: {avg_loss:.4f}")
    print(f"Standard Deviation of Validation Loss: {std_loss:.4f}")

    return 0
