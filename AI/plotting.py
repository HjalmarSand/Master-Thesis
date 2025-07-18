from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


def create_cross_match_plot(
    query_image_path: Path, top_matches: list, dem_mask_db_path: Path, save_path: Path
):
    """
    Creates a plot showing the query image and its top N DEM skyline matches.
    """
    num_matches = len(top_matches)
    fig, axs = plt.subplots(
        num_matches + 1,
        1,
        figsize=(12, 3 * (num_matches + 1)),
        gridspec_kw={"height_ratios": [1.5] + [1] * num_matches},
    )

    # --- Plot the Query Image ---
    with Image.open(query_image_path) as img:
        axs[0].imshow(np.array(img), cmap="gray")
    axs[0].set_title(f"Query: {query_image_path.name}", fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # --- Plot the Top DEM Matches ---
    with np.load(dem_mask_db_path) as dem_data:
        for i, match in enumerate(top_matches):
            ax = axs[i + 1]
            match_key = match["match_key"]

            if match_key in dem_data:
                mask_array = dem_data[match_key]
                ax.imshow(mask_array, cmap="gray")

            title = (
                f"Match #{i + 1}: {match_key}\n"
                f"Score: {match['score']:.4f} | Error: {match['error_m']:.2f} m"
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
