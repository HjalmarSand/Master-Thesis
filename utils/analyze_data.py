import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The specific path is no longer needed, as the function will find all dataset files.
# GPS_CSV_PATH = "images/dataset1/gps_interpolated_with_orientation.csv"
OUTPUT_DIR = "trajectory_analysis_output"


def calculate_distance_easting_northing(e1, n1, e2, n2):
    """Calculates distance between two Easting/Northing points."""
    return math.sqrt((e2 - e1) ** 2 + (n2 - n1) ** 2)


def calculate_angle_difference(angle1, angle2):
    """Calculates the shortest difference between two angles (0-360 degrees)."""
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff


def extract_numeric_from_filename(filename):
    """Extracts the numeric part from filenames like '0801.jpg' for sorting."""
    try:
        return int(os.path.splitext(os.path.basename(filename))[0])
    except ValueError:
        return np.nan


def analyze_trajectory(base_search_path="images"):
    """
    Finds and analyzes trajectory data for all datasets in a base directory.

    Searches for 'images/dataset*/gps_interpolated_with_orientation.csv',
    prints statistics for each, and plots all trajectories on combined graphs.
    """
    search_pattern = os.path.join(
        base_search_path, "dataset*", "gps_interpolated_with_orientation.csv"
    )
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"Error: No trajectory CSV files found with pattern: {search_pattern}")
        return

    print(f"Found {len(csv_files)} trajectory files to analyze: {csv_files}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Set up the plots to hold all trajectories ---
    fig_traj, ax_traj = plt.subplots(figsize=(12, 10))
    fig_orient, ax_orient = plt.subplots(figsize=(12, 10))

    # Generate a list of distinct colors for each trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

    # --- Loop through each found CSV file ---
    for i, csv_path in enumerate(csv_files):
        dataset_name = os.path.basename(os.path.dirname(csv_path))
        print(f"\n{'=' * 20} Analyzing: {dataset_name} {'=' * 20}")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            continue  # Skip to the next file

        required_cols = ["filename", "easting", "northing", "Orientation"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: CSV in {dataset_name} must contain columns: {', '.join(required_cols)}"
            )
            print(f"Found columns: {', '.join(df.columns)}")
            continue

        # Convert coordinates to numeric, coercing errors to NaN
        df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
        df["northing"] = pd.to_numeric(df["northing"], errors="coerce")
        df["Orientation"] = pd.to_numeric(df["Orientation"], errors="coerce")

        df.dropna(subset=["easting", "northing", "Orientation"], inplace=True)

        if df.empty:
            print(f"Error: No valid data in {dataset_name} after cleaning NaNs.")
            continue

        df["sort_key"] = df["filename"].apply(extract_numeric_from_filename)
        df.dropna(subset=["sort_key"], inplace=True)
        df.sort_values(by="sort_key", inplace=True)
        df.drop(columns=["sort_key"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 2:
            print(
                f"Not enough data points (< 2) in {dataset_name} to analyze trajectory."
            )
            continue

        print(
            f"Successfully loaded and sorted {len(df)} data points for {dataset_name}."
        )

        # --- Calculate Statistics for the current trajectory ---
        distances = []
        orientation_changes = []

        for j in range(1, len(df)):
            dist = calculate_distance_easting_northing(
                df["easting"].iloc[j - 1],
                df["northing"].iloc[j - 1],
                df["easting"].iloc[j],
                df["northing"].iloc[j],
            )
            distances.append(dist)

            orient_change = calculate_angle_difference(
                df["Orientation"].iloc[j - 1], df["Orientation"].iloc[j]
            )
            orientation_changes.append(orient_change)

        # Print Individual Statistics
        print("\n--- Trajectory Statistics ---")
        total_distance_m = sum(distances) if distances else 0
        avg_distance_per_segment_m = np.mean(distances) if distances else 0

        print(f"Total trajectory length: {total_distance_m:.2f} meters")
        print(f"Average distance between images: {avg_distance_per_segment_m:.2f} m")
        # (Add more stats printing here if desired)

        # --- Plotting the current trajectory on the shared axes ---
        color = colors[i]

        # Plot on the main trajectory figure
        ax_traj.plot(
            df["easting"],
            df["northing"],
            marker="o",
            linestyle="-",
            markersize=2,
            label=dataset_name,
            color=color,
        )
        # Mark start (square) and end (X) points
        ax_traj.plot(
            df["easting"].iloc[0],
            df["northing"].iloc[0],
            marker="s",
            markersize=8,
            color=color,
            label="_nolegend_",
        )
        ax_traj.plot(
            df["easting"].iloc[-1],
            df["northing"].iloc[-1],
            marker="X",
            markersize=10,
            color=color,
            label="_nolegend_",
        )

        # Plot on the orientation figure
        orientation_scale = (
            avg_distance_per_segment_m * 0.8 if avg_distance_per_segment_m > 0 else 1.0
        )
        angles_rad = np.deg2rad(90 - df["Orientation"])
        U = orientation_scale * np.cos(angles_rad)
        V = orientation_scale * np.sin(angles_rad)

        ax_orient.quiver(
            df["easting"],
            df["northing"],
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            headwidth=3,
            headlength=4,
            color=color,
            label=dataset_name,
        )
        ax_orient.plot(
            df["easting"].iloc[0],
            df["northing"].iloc[0],
            marker="s",
            markersize=8,
            color=color,
            label="_nolegend_",
        )
        ax_orient.plot(
            df["easting"].iloc[-1],
            df["northing"].iloc[-1],
            marker="X",
            markersize=10,
            color=color,
            label="_nolegend_",
        )

    # --- Finalize and Save the Combined Plots ---
    print("\nFinalizing combined plots...")

    # Finalize Trajectory Plot
    ax_traj.set_title("Combined Trajectory Plot")
    ax_traj.set_xlabel("Easting (m)")
    ax_traj.set_ylabel("Northing (m)")
    ax_traj.legend(title="Datasets")
    ax_traj.grid(True)
    ax_traj.axis("equal")
    traj_save_path = os.path.join(OUTPUT_DIR, "combined_trajectory_plot.png")
    fig_traj.savefig(traj_save_path, dpi=300, bbox_inches="tight")
    print(f"Combined trajectory plot saved to {traj_save_path}")

    # Finalize Orientation Plot
    ax_orient.set_title("Combined Trajectory with Orientation Arrows")
    ax_orient.set_xlabel("Easting (m)")
    ax_orient.set_ylabel("Northing (m)")
    ax_orient.legend(title="Datasets")
    ax_orient.grid(True)
    ax_orient.axis("equal")
    orient_save_path = os.path.join(
        OUTPUT_DIR, "combined_trajectory_orientation_plot.png"
    )
    fig_orient.savefig(orient_save_path, dpi=300, bbox_inches="tight")
    print(f"Combined orientation plot saved to {orient_save_path}")

    print("\nAnalysis complete. All plots saved in", OUTPUT_DIR)


def analyze_and_plot_results(directories=None, directory="."):
    """
    Finds all 'output*.csv' files in one or more directories, calculates adjusted average error,
    and plots the results.

    Args:
        directories: List of directory paths or dictionary with {label: path} pairs
                    If None, uses the single 'directory' parameter
        directory: Single directory path (used if directories is None)
    """

    ERROR_ADJUSTMENT = 9.30  # Average length between two GPS points in the dataset
    CSV_FILE_PATTERN = "output*.csv"  # Pattern to match relevant CSV files

    # Handle different input formats
    if directories is None:
        # Single directory mode (backward compatibility)
        directories = {"Results": directory}
    elif isinstance(directories, list):
        # Convert list to dictionary with automatic labels
        directories = {f"Folder {i + 1}": path for i, path in enumerate(directories)}
    elif not isinstance(directories, dict):
        print(
            "Error: 'directories' must be a list of paths or a dictionary of {label: path}"
        )
        return

    # Define colors for different curves
    colors = ["b", "r", "g", "c", "m", "y", "k", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "+", "x"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    all_data_found = False

    # Process each directory
    for idx, (label, dir_path) in enumerate(directories.items()):
        print(f"\nProcessing directory: {label} ({dir_path})")

        # Discover all relevant CSV files in the specified directory
        search_path = os.path.join(dir_path, CSV_FILE_PATTERN)
        csv_files = glob.glob(search_path)

        if not csv_files:
            print(
                f"  Warning: No files found matching the pattern '{CSV_FILE_PATTERN}' in {dir_path}"
            )
            continue

        print(f"  Found {len(csv_files)} result files to analyze.")

        plot_data = []

        # Process each file to extract radius and calculate adjusted error
        for filepath in csv_files:
            filename = os.path.basename(filepath)

            # Use a regular expression to reliably extract the numeric radius from the filename
            match = re.search(r"output(\d+)\.csv", filename)
            if not match:
                print(
                    f"  Warning: Skipping file with unexpected name format: {filename}"
                )
                continue

            radius = int(match.group(1))

            try:
                df = pd.read_csv(filepath)

                # Calculate the average error
                average_error = df["distance_meters"].mean()

                # Since the average error doesnt take the radius exclusion into account, we adjust the average error
                adjusted_average_error = average_error - ERROR_ADJUSTMENT * (radius + 1)

                plot_data.append(
                    {"radius": radius, "adjusted_error": adjusted_average_error}
                )

                print(
                    f"    - Processed {filename}: Radius={radius}, Adjusted Avg Error={adjusted_average_error:.2f}m"
                )

            except pd.errors.EmptyDataError:
                print(f"  Warning: The file {filename} is empty. Skipping.")
            except Exception as e:
                print(f"  An error occurred while processing {filename}: {e}")

        if not plot_data:
            print(f"  No data could be processed for {label}.")
            continue

        all_data_found = True

        # Sort the data by radius to ensure the plot is drawn in a logical order
        plot_data.sort(key=lambda item: item["radius"])

        y_values = [item["adjusted_error"] for item in plot_data]
        x_values = [item["radius"] * 9.30 for item in plot_data]

        # Choose color and marker
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Plot the curve for this directory
        ax.plot(
            x_values,
            y_values,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
            markersize=8,
            linewidth=2,
        )

    if not all_data_found:
        print("\nNo data could be processed from any directory.")
        return

    # Finalize the plot
    ax.set_ylabel("Error (m)", fontsize=12)
    ax.set_xlabel("Exclusion radius (m)", fontsize=12)
    ax.set_title(
        "Impact of Exclusion Radius on Localization Error", fontsize=16, weight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, "multi_folder_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    # --- Run Trajectory Analysis ---
    # This will now find all 'dataset*' folders inside 'images' and plot them together.
    analyze_trajectory()

    # --- Run Error vs. Exclusion Radius Analysis ---
    # This part remains unchanged and can be used as before.
    # Example usage for multiple folders:
    # Option 1: Using a list of directories (auto-labeled)
    # analyze_and_plot_results(directories=["results_1", "results_2", "results_3"])

    # Option 2: Using a dictionary with custom labels
    # analyze_and_plot_results(
    #     directories={
    #         "Distorted": "results/distorted",
    #         "Undistorted": "results/undistorted",
    #     }
    # )

    # Option 3: Single directory (backward compatible)
    # analyze_and_plot_results(directory="results")
