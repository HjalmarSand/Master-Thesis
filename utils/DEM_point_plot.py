import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pdal
from mpl_toolkits.mplot3d import Axes3D

LAZ_DIRECTORY = Path("./DSM_Data/cleaned_laser")
VIEWPOINT_COORDS = (613201, 6461722, 2.0)  # (X, Y, Z)

SEARCH_RADIUS = 1000.0

# Set to a file path to save the plot, or None to display it interactively
OUTPUT_FILENAME = "point_cloud_view.png"
# Dots per meter for the saved image
DPI_SETTING = 300
POINT_SIZE = 0.5
COLORMAP = "terrain"


def find_relevant_files(
    directory: Path, viewpoint: Tuple[float, float], radius: float
) -> List[Path]:
    """
    Finds LAZ files whose bounding boxes intersect with the search area.
    This is an optimization to avoid opening every single file.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_laz_files = sorted(
        list(directory.glob("*.laz")) + list(directory.glob("*.LAZ"))
    )
    if not all_laz_files:
        raise FileNotFoundError(f"No LAZ files found in {directory}")

    relevant_files = []
    view_x, view_y = viewpoint

    print(
        f"Scanning {len(all_laz_files)} files to find those relevant to the search area..."
    )

    for laz_file in all_laz_files:
        # For simplicity in a non-interactive script, we can skip the JSON check
        # and just try to load data from each.
        # Here, we'll include all files and let the PDAL pipeline do the cropping.
        # This is simpler but might be slower for many files.
        relevant_files.append(laz_file)

    print(f"Found {len(relevant_files)} potentially relevant files to process.")
    return relevant_files


def load_points_near_viewpoint(
    laz_files: List[Path], viewpoint: Tuple[float, float, float], radius: float
) -> Optional[np.ndarray]:
    """
    Loads point cloud data from a list of LAZ files that fall within a
    circular radius of a given viewpoint.

    Args:
        laz_files: List of LAZ file paths to process.
        viewpoint: An (X, Y, Z) tuple for the center of the search area.
        radius: The search radius.

    Returns:
        A NumPy structured array containing all loaded points, or None if no points are found.
    """
    all_points_list = []
    view_x, view_y, _ = viewpoint

    # Define a square cropping boundary for the PDAL pipeline for efficiency
    crop_bounds = f"([{view_x - radius}, {view_x + radius}], [{view_y - radius}, {view_y + radius}])"

    for laz_file in laz_files:
        try:
            print(f"Processing {laz_file.name}...")
            pipeline_json = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(laz_file)},
                    {"type": "filters.crop", "bounds": crop_bounds},
                ]
            }

            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            count = pipeline.execute()

            if count > 0:
                points = pipeline.arrays[0]
                # Further filter from the square crop to a circular radius
                distances = np.sqrt(
                    (points["X"] - view_x) ** 2 + (points["Y"] - view_y) ** 2
                )
                within_radius_mask = distances <= radius

                filtered_points = points[within_radius_mask]

                if len(filtered_points) > 0:
                    all_points_list.append(filtered_points)
                    print(f"  -> Added {len(filtered_points):,} points.")

        except Exception as e:
            print(f"  Could not process {laz_file.name}: {e}")

    if not all_points_list:
        return None

    # Combine all collected points into a single NumPy array
    combined_points = np.concatenate(all_points_list)
    return combined_points


def create_3d_plot(points: np.ndarray, viewpoint: Tuple[float, float, float]):
    """Creates and styles a 3D matplotlib plot of the point cloud."""

    print("\nGenerating 3D plot...")
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of the point cloud, colored by elevation (Z-value)
    scatter = ax.scatter(
        points["X"],
        points["Y"],
        points["Z"],
        s=POINT_SIZE,  # Point size
        c=points["Z"],  # Color by elevation
        cmap=COLORMAP,  # Colormap for elevation
        alpha=0.8,
    )

    # Plot the viewpoint as a distinct marker
    ax.scatter(
        viewpoint[0],
        viewpoint[1],
        viewpoint[2],
        c="red",
        s=150,
        marker="X",
        depthshade=True,  # Make sure marker is always visible
        label="Viewpoint",
    )

    # Add a color bar to show the elevation scale
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Elevation (Z)")

    # Set labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate (Elevation)")
    ax.set_title(f"Point Cloud within {SEARCH_RADIUS:.0f}m of Viewpoint")

    # Improve aspect ratio to be more representative of the data's scale
    # This prevents the plot from looking overly stretched on one axis.
    x_range = np.ptp(points["X"])
    y_range = np.ptp(points["Y"])
    z_range = np.ptp(points["Z"])
    ax.set_box_aspect([x_range, y_range, z_range])

    # Set a good initial camera angle
    ax.view_init(elev=45.0, azim=-75)

    ax.legend()
    plt.tight_layout()

    return fig


def main():
    """Main execution function."""
    print("--- Static LAZ Point Cloud Plotter ---")
    try:
        relevant_files = find_relevant_files(
            LAZ_DIRECTORY, VIEWPOINT_COORDS[:2], SEARCH_RADIUS
        )
        if not relevant_files:
            print("No relevant files found to process.")
            return

        # Load points from those files that are within the search radius
        points_data = load_points_near_viewpoint(
            relevant_files, VIEWPOINT_COORDS, SEARCH_RADIUS
        )

        if points_data is None or len(points_data) == 0:
            print("\nNo points found within the specified radius. Exiting.")
            return

        print(f"\nTotal points loaded: {len(points_data):,}")

        # Create the 3D plot
        fig = create_3d_plot(points_data, VIEWPOINT_COORDS)

        # 4. Save or show the plot
        if OUTPUT_FILENAME:
            print(f"Saving plot to {OUTPUT_FILENAME}...")
            fig.savefig(OUTPUT_FILENAME, dpi=DPI_SETTING, bbox_inches="tight")
            print("Save complete.")
        else:
            print("Displaying plot...")
            plt.show()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
