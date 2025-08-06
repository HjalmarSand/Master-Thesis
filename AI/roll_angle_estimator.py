import random
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from tqdm import tqdm

from utils.utility import get_original_numbered_file_name


# 1. Custom PyTorch Dataset
class RollAngleDataset(Dataset):
    """
    A robust Dataset that correctly handles augmentations for a
    regression task by coupling image and label transformations.
    """

    def __init__(
        self,
        training_sources: Dict[Path, str],
        annotations_csv: Path,
        mode: str = "train",  # Can be 'train' or 'eval'
        resize_dim: int = 256,
        crop_dim: int = 224,
    ):
        annotations_df = pd.read_csv(annotations_csv)
        self.data_points = []
        print(f"\nBuilding Combined Dataset in '{mode}' mode")
        for image_dir, angle_column_name in training_sources.items():
            specific_annotations_df = annotations_df.dropna(subset=[angle_column_name])
            angle_lookup = {
                (row.dataset, row.index): getattr(row, angle_column_name)
                for row in specific_annotations_df.itertuples(index=False)
            }
            if not angle_lookup:
                continue
            try:
                dataset_num = int(re.search(r"(\d+)", image_dir.parent.name).group(1))
            except (AttributeError, ValueError):
                continue
            image_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_paths.extend(image_dir.glob(ext))
            for path in sorted(image_paths):
                match = re.search(r"(\d{4})", path.name)
                if match:
                    lookup_key = (dataset_num, int(match.group(1)))
                    if lookup_key in angle_lookup:
                        self.data_points.append((path, angle_lookup[lookup_key]))

        if not self.data_points:
            raise ValueError(
                "Dataset construction failed: No images matched any annotations."
            )
        print(f"Successfully built dataset with {len(self.data_points)} items.")

        self.mode = mode
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        img_path, roll_angle = self.data_points[index]
        image = Image.open(img_path).convert("L")

        # AUGMENTATION LOGIC
        # Apply augmentations only in 'train' mode
        if self.mode == "train":
            # 1. Get a random rotation angle
            rotation_angle = random.uniform(-15, 15)

            # 2. Apply the rotation to the image
            image = TF.rotate(image, angle=rotation_angle)

            # 3. CRUCIAL FIX: Correct the label by the same amount
            roll_angle += rotation_angle

            # Apply other non-geometric augmentations
            if random.random() > 0.5:
                image = TF.adjust_brightness(
                    image, brightness_factor=random.uniform(0.7, 1.3)
                )
            if random.random() > 0.5:
                image = TF.adjust_contrast(
                    image, contrast_factor=random.uniform(0.7, 1.3)
                )

        # STANDARD TRANSFORMS (for both train and eval)
        image = TF.resize(image, [self.resize_dim])
        image = TF.center_crop(image, [self.crop_dim, self.crop_dim])
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # FOCUS MASK
        _, height, width = image.shape
        center_x, center_y = width // 2, height // 2
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        scale = width / 2.5
        dist_from_center = torch.sqrt(
            ((X - center_x) ** 2 + (Y - center_y) ** 2).float()
        )
        focus_mask = torch.exp(-(dist_from_center**2) / (2 * scale**2))
        image = image * focus_mask.unsqueeze(0).to(image.device)

        return (image, torch.tensor(float(roll_angle)))


# 2. Simple CNN Model for Regression
class RollEstimatorCNN(nn.Module):
    def __init__(self):
        super(RollEstimatorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
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
    training_sources: Dict[Path, str],
    annotation_csv: Path,
    model_save_path: Path,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    print("\nStarting Roll Angle Estimator Training")
    print(f"Using data from {len(training_sources)} source(s).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # We no longer define transforms here, the Dataset handles it.

    # 1. Create a temporary dataset just to determine the total size and split indices
    # We use 'eval' mode as it's faster and doesn't matter for index splitting.
    dataset_for_splitting = RollAngleDataset(
        annotations_csv=annotation_csv, training_sources=training_sources, mode="eval"
    )

    # 2. Split the indices into training and validation sets
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset_for_splitting))
    val_size = len(dataset_for_splitting) - train_size
    train_subset, val_subset = random_split(
        dataset_for_splitting, [train_size, val_size], generator=generator
    )

    # 3. Create the actual datasets for training and validation with the correct modes
    train_dataset = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="train",  # Use 'train' mode to enable augmentations
    )

    val_dataset = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="eval",  # Use 'eval' mode to disable augmentations
    )

    # 4. Create samplers to draw from the correct index splits for each dataset
    train_sampler = SubsetRandomSampler(train_subset.indices)
    val_sampler = SubsetRandomSampler(val_subset.indices)

    print(f"Total training set size: {len(train_sampler)}")
    print(f"Total validation set size: {len(val_sampler)}")

    # 5. Create DataLoaders using the datasets and their respective samplers
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, sampler=val_sampler
    )

    model = RollEstimatorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    best_val_loss = float("inf")
    best_model_path = model_save_path.with_name(f"{model_save_path.stem}_best.pth")

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
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Validation loss improved. Saved best model to {best_model_path.name}"
            )

    print(
        f"\nTraining complete. Best model saved to {best_model_path.name} with validation loss: {best_val_loss:.4f}"
    )
    return 0


def predict_angles_for_directory(
    image_dir: Path, model_path: Path, output_csv_path: Path
):
    """
    Loads a trained model and predicts roll angles for all images in a directory,
    saving the results to a CSV file.
    """
    print(f"\nPredicting Roll Angles for {image_dir.parent.name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not model_path.exists():
        print(f"ERROR: Model file not found at '{model_path}'.")
        return 1

    model = RollEstimatorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded trained model from '{model_path.name}'")

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not image_paths:
        print(f"No images found in {image_dir}")
        return 1

    resize_dim = 256
    crop_dim = 224

    results = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting Angles"):
            image = Image.open(img_path).convert("L")

            # Manually apply the same transforms as 'eval' mode in the dataset
            tensor = TF.resize(image, [resize_dim])
            tensor = TF.center_crop(tensor, [crop_dim, crop_dim])
            tensor = TF.to_tensor(tensor)
            tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])

            # Apply the focus mask
            _, height, width = tensor.shape
            center_x, center_y = width // 2, height // 2
            Y, X = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            scale = width / 2.5
            dist_from_center = torch.sqrt(
                ((X - center_x) ** 2 + (Y - center_y) ** 2).float()
            )
            focus_mask = torch.exp(-(dist_from_center**2) / (2 * scale**2))
            tensor = tensor * focus_mask.unsqueeze(0)

            tensor = tensor.unsqueeze(0).to(device)

            predicted_angle = model(tensor).item()
            results.append(
                {
                    "dataset": image_dir.parent.name,
                    "filename": get_original_numbered_file_name(img_path.name)[:-4],
                    "predicted_roll": predicted_angle,
                }
            )

    df = pd.DataFrame(results)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, float_format="%.4f")
    print(f"Predictions saved to {output_csv_path}")
    return 0


def visualize_model_features(model_path: Path, image_path: Path, output_dir: Path):
    """
    Loads a trained model and visualizes its first-layer filters and the
    feature maps for a given sample image.
    """
    print("\nVisualizing Model Features")
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

    resize_dim = 256
    crop_dim = 224

    image = Image.open(image_path).convert("L")
    tensor = TF.resize(image, [resize_dim])
    tensor = TF.center_crop(tensor, [crop_dim, crop_dim])
    tensor = TF.to_tensor(tensor)
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
    tensor = tensor.unsqueeze(0).to(device)

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


def run_kfold_cross_validation(
    training_sources: Dict[Path, str],
    annotation_csv: Path,
    k_folds: int = 5,
    num_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    Performs K-Fold Cross-Validation using the corrected Dataset logic,
    ensuring augmentations are only applied to the training split of each fold.
    """
    print(f"\nStarting {k_folds}-Fold Cross-Validation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create one dataset instance just to get the indices for splitting
    dataset_for_splitting = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="eval",  # Mode doesn't matter here, just for getting size
    )

    # Create the two dataset instances that will actually be used
    train_dataset_kfold = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="train",  # This dataset will apply augmentations
    )
    val_dataset_kfold = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="eval",  # This dataset will NOT apply augmentations
    )

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}

    # kfold.split gives us INDICES, which we use to sample from our datasets
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_for_splitting)):
        print("\n" + "=" * 50)
        print(f"Fold {fold + 1} / {k_folds}")
        print("=" * 50)

        # Sample from the 'train' dataset for the training loader
        train_sampler = SubsetRandomSampler(train_ids)
        train_loader = DataLoader(
            train_dataset_kfold, batch_size=batch_size, sampler=train_sampler
        )

        # Sample from the 'eval' dataset for the validation loader
        val_sampler = SubsetRandomSampler(val_ids)
        val_loader = DataLoader(
            val_dataset_kfold, batch_size=batch_size, sampler=val_sampler
        )

        model = RollEstimatorCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # The inner training loop remains the same
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


def analyze_model_performance(
    training_sources: Dict[Path, str],
    annotation_csv: Path,
    model_path: Path,
    results_dir: Path,
):
    """
    Loads the best model and performs a detailed analysis on the validation set.
    """
    print("\nAnalyzing Model Performance on Validation Set")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Recreate the exact validation set
    # Use 'eval' mode as it's faster and doesn't matter for index splitting.
    dataset_for_splitting = RollAngleDataset(
        annotations_csv=annotation_csv, training_sources=training_sources, mode="eval"
    )

    train_size = int(0.8 * len(dataset_for_splitting))
    val_size = len(dataset_for_splitting) - train_size
    # We must use a fixed generator to get the same split every time
    generator = torch.Generator().manual_seed(42)
    _, val_subset = random_split(
        dataset_for_splitting, [train_size, val_size], generator=generator
    )

    val_dataset = RollAngleDataset(
        annotations_csv=annotation_csv,
        training_sources=training_sources,
        mode="eval",  # Use eval mode for clean data
    )

    # We only need the validation sampler and loader
    val_sampler = SubsetRandomSampler(val_subset.indices)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

    model = RollEstimatorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Get predictions
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for images, angles in tqdm(val_loader, desc="Getting Predictions"):
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_trues.extend(angles.cpu().numpy().flatten())

    # 4. Calculate and print metrics
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    mse = np.mean((all_trues - all_preds) ** 2)
    mae = np.mean(np.abs(all_trues - all_preds))

    print(f"\nPerformance on {len(val_subset)} validation images:")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f} degrees")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f} degrees")

    # 5. Create and save a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(all_trues, all_preds, alpha=0.5, label="Predictions")
    plt.plot([-20, 20], [-20, 20], "r--", label="Ideal Prediction")
    plt.title("True vs. Predicted Roll Angles")
    plt.xlabel("True Angle (degrees)")
    plt.ylabel("Predicted Angle (degrees)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_save_path = results_dir / "evaluation_scatter_plot.png"
    plt.savefig(plot_save_path)
    print(f"✅ Saved evaluation plot to {plot_save_path}")

    return 0
