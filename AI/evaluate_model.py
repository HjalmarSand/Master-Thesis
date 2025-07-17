# AI/evaluate_model.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split

# We need to import the model and dataset class definitions
from .roll_angle_estimator import RollAngleDataset, RollEstimatorCNN


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> tuple:
    """
    Runs the model on the provided data loader and collects predictions and true labels.
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, angles in data_loader:
            images = images.to(device)
            # Make predictions
            outputs = model(images)

            # Move data to CPU and store it
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_true_labels.extend(angles.numpy())

    return np.array(all_true_labels), np.array(all_predictions)


def create_results_plot(
    true_labels: np.ndarray, predictions: np.ndarray, save_path: Path
):
    """
    Creates a scatter plot of true vs. predicted values.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(true_labels, predictions, alpha=0.5, label="Predictions")

    # Plot a perfect-fit line (y=x) for reference
    perfect_fit_line = np.linspace(min(true_labels), max(true_labels), 100)
    plt.plot(perfect_fit_line, perfect_fit_line, "r--", label="Perfect Fit")

    plt.title("Model Evaluation: True vs. Predicted Roll Angles")
    plt.xlabel("True Roll Angle (degrees)")
    plt.ylabel("Predicted Roll Angle (degrees)")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    # Save the plot to a file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Evaluation plot saved to: {save_path}")
    plt.close()


def run_evaluation(
    images_root_dir: Path,
    annotation_csv: Path,
    model_path: Path,
    plot_save_path: Path,
    transform,
):
    """
    Main function to orchestrate the model evaluation process.
    """
    print("\n--- Starting Model Evaluation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # We create the full dataset and split it with the same logic as in training
    # to ensure we are using the exact same validation set.
    full_dataset = RollAngleDataset(
        csv_file=annotation_csv, images_root_dir=images_root_dir, transform=transform
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Using a fixed generator seed ensures the split is identical every time
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=32)
    print(f"Loaded {len(val_dataset)} images for validation.")

    model = RollEstimatorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Successfully loaded trained model from: {model_path}")

    true_angles, pred_angles = evaluate_model(model, val_loader, device)

    mae = mean_absolute_error(true_angles, pred_angles)
    rmse = np.sqrt(mean_squared_error(true_angles, pred_angles))

    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.3f} degrees")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} degrees")
    print("--------------------------")
    print("MAE means: On average, the model's prediction is off by this many degrees.")
    print("RMSE punishes larger errors more heavily.")

    # --- Create and Save Plot ---
    create_results_plot(true_angles, pred_angles, plot_save_path)

    return 0
