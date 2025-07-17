from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_results_plot(
    true_angles: np.ndarray, pred_angles: np.ndarray, save_path: Path
):
    """
    Creates and saves a scatter plot of true vs. predicted angles.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create the scatter plot
    ax.scatter(true_angles, pred_angles, alpha=0.5, label="Predictions")

    # Add a diagonal y=x line for reference (perfect prediction)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect Prediction")

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True Roll Angle (degrees)")
    ax.set_ylabel("Predicted Roll Angle (degrees)")
    ax.set_title("True vs. Predicted Roll Angle")
    ax.legend()
    ax.grid(True)

    # Save the figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)  # Close the plot to free memory

    print(f"Evaluation plot saved to: {save_path}")
