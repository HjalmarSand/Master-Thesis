# In your evaluation script (e.g., AI/evaluate_model.py)

from pathlib import Path
from typing import List  # Make sure to import List

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split

from .plotting import create_results_plot  # Assuming you have this helper

# Import your updated classes and functions
from .roll_angle_estimator import RollAngleDataset, RollEstimatorCNN


def evaluate_model(model, data_loader, device):
    """Evaluates the model on the data from the data loader."""
    model.eval()
    all_true_angles = []
    all_pred_angles = []
    with torch.no_grad():
        for images, angles in data_loader:
            images = images.to(device)
            angles = angles.to(device).float().unsqueeze(1)

            outputs = model(images)

            all_true_angles.extend(angles.cpu().numpy().flatten())
            all_pred_angles.extend(outputs.cpu().numpy().flatten())

    return np.array(all_true_angles), np.array(all_pred_angles)


def run_evaluation(
    image_dirs: List[Path],
    annotation_csv: Path,
    model_path: Path,
    plot_save_path: Path,
    transform,
):
    """
    Main function to orchestrate the model evaluation process and find worst predictions.
    """
    print("\n--- Starting Model Evaluation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_dataset = RollAngleDataset(
        annotations_csv=annotation_csv, image_dirs=image_dirs, transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=32)
    print(f"Evaluating on the validation set of {len(val_dataset)} images.")

    model = RollEstimatorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    true_angles, pred_angles = evaluate_model(model, val_loader, device)

    mae = mean_absolute_error(true_angles, pred_angles)
    rmse = np.sqrt(mean_squared_error(true_angles, pred_angles))

    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.3f} degrees")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} degrees")

    create_results_plot(true_angles, pred_angles, plot_save_path)

    print("\n--- Top 5 Worst Predictions ---")
    errors = np.abs(true_angles - pred_angles)
    sorted_indices = np.argsort(errors)
    worst_indices = sorted_indices[-5:]

    for i in reversed(worst_indices):
        original_data_index = val_dataset.indices[i]
        img_path, _ = full_dataset.data_points[original_data_index]

        # Get the dataset name from the image path
        dataset_name = img_path.parent.parent.name

        true_val = true_angles[i]
        pred_val = pred_angles[i]
        error_val = errors[i]

        # Updated print statement
        print(
            f"Dataset: {dataset_name:<10} | "
            f"File: {img_path.name:<30} | "
            f"True: {true_val:>6.2f}, "
            f"Pred: {pred_val:>6.2f}, "
            f"Error: {error_val:>5.2f}"
        )

    return 0
