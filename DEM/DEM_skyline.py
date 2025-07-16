import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
from scipy.signal import medfilt
from tqdm import tqdm

import constants

# This global variable will be initialized in each worker process
global_constants = {}


def find_laz_files() -> List[Path]:
    """Find all LAZ files in the configured directory."""
    laz_directory = Path(constants.LIDAR_DATA_DIR)
    print(f"Searching for LAZ files in directory: {laz_directory}")
    if not laz_directory.exists():
        raise FileNotFoundError(f"Directory not found: {laz_directory}")
    laz_files = sorted(
        list(laz_directory.glob("*.laz")) + list(laz_directory.glob("*.LAZ"))
    )
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in directory: {laz_directory}")
    print(f"Found {len(laz_files)} LAZ files.")
    return laz_files


def check_file_relevance_from_json(
    file_path: Path, viewpoint_coords: Tuple[float, float, float]
) -> bool:
    """
    DEBUG VERSION: Checks if a LAZ file's bounds are relevant to the viewpoint
    and prints its reasoning.
    """
    vp_x, vp_y, _ = viewpoint_coords
    new_name = f"{file_path.stem}_tile.json"
    json_path = file_path.with_name(new_name)

    # Print the values we're using for the check
    search_radius = constants.SEARCH_RADIUS

    try:
        if not json_path.exists():
            return True

        with open(json_path, "r") as f:
            data = json.load(f)

        coords = data["features"][0]["geometry"]["coordinates"][0][:-1]
        x_coords, y_coords = [c[0] for c in coords], [c[1] for c in coords]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        closest_x = max(min_x, min(vp_x, max_x))
        closest_y = max(min_y, min(vp_y, max_y))
        min_dist = np.sqrt((vp_x - closest_x) ** 2 + (vp_y - closest_y) ** 2)

        is_relevant = min_dist <= search_radius

        return is_relevant

    except Exception as e:
        return True


def generate_skyline_mask_array(
    angles: np.ndarray, elevations: np.ndarray, output_width: int, output_height: int
) -> np.ndarray:
    """Creates a binary mask array from aggregated skyline data, with median filtering."""
    sort_indices = np.argsort(angles)
    sorted_angles, sorted_elevations = angles[sort_indices], elevations[sort_indices]

    # Wrap angles and elevations to ensure seamless interpolation at the 0/360 degree boundary
    wrapped_angles = np.concatenate(
        ([sorted_angles[-1] - 360], sorted_angles, [sorted_angles[0] + 360])
    )
    wrapped_elevations = np.concatenate(
        ([sorted_elevations[-1]], sorted_elevations, [sorted_elevations[0]])
    )

    output_azimuths = np.linspace(0, 360, output_width, endpoint=False)
    interpolated_elevations = np.interp(
        output_azimuths, wrapped_angles, wrapped_elevations, period=360
    )

    # Smooth the skyline profile to reduce noise
    filtered_elevations = medfilt(interpolated_elevations, kernel_size=19)

    # Roll the array to align with a standard panorama projection (center is 0 degrees azimuth)
    pixels_to_roll = -int(output_width / 2)
    final_elevations = np.roll(filtered_elevations, pixels_to_roll)

    # Convert elevation angles to pixel rows
    v_fov = 90.0  # Vertical Field of View (-45 to +45 degrees)
    top_elevation = v_fov / 2.0
    skyline_rows = np.clip(
        np.round(((top_elevation - final_elevations) / v_fov) * output_height),
        0,
        output_height,
    ).astype(int)

    # Create the final binary mask
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    for x, y_sky in enumerate(skyline_rows):
        mask[0:y_sky, x] = 255  # 255 represents the sky

    return mask


def init_worker(config: dict):
    """Initializes each worker process with shared configuration."""
    global global_constants
    global_constants = config


def process_viewpoint_and_file(task: Tuple) -> Tuple:
    """
    Worker function that processes a SINGLE LAZ file for a SINGLE viewpoint.
    This is the core unit of work for the parallel processing pool.
    """
    point_id, viewpoint_coords, file_path = task
    try:
        # Load constants provided by the init_worker function
        search_radius = global_constants["SEARCH_RADIUS"]
        min_distance = global_constants["MIN_DISTANCE"]

        # Define the bounding box for the PDAL crop filter
        vp_x, vp_y, vp_z = viewpoint_coords
        min_x, max_x = vp_x - search_radius, vp_x + search_radius
        min_y, max_y = vp_y - search_radius, vp_y + search_radius

        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(file_path)},
                {
                    "type": "filters.crop",
                    "bounds": f"([{min_x}, {max_x}],[{min_y}, {max_y}])",
                },
            ]
        }
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        if pipeline.execute() == 0:
            return (point_id, None)  # No points read

        points = pipeline.arrays[0]
        if len(points) == 0:
            return (point_id, None)

        # Calculate coordinates relative to the viewpoint
        rel_x = points["X"] - vp_x
        rel_y = points["Y"] - vp_y
        rel_z = points["Z"] - vp_z

        horizontal_dist = np.sqrt(rel_x**2 + rel_y**2)
        valid_mask = horizontal_dist > min_distance

        if not np.any(valid_mask):
            return (point_id, None)

        # Calculate azimuth and elevation for each valid point
        azimuth_deg = (
            np.degrees(np.arctan2(rel_x[valid_mask], rel_y[valid_mask])) + 360
        ) % 360
        elevation_deg = np.degrees(
            np.arctan2(rel_z[valid_mask], horizontal_dist[valid_mask])
        )

        # Return the raw data for aggregation in the main process
        return (point_id, (azimuth_deg, elevation_deg))

    except Exception:
        # Silently fail to avoid flooding console with errors from many workers
        return (point_id, None)


def create_and_save_grid(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    spacing_meters: float,
) -> pd.DataFrame:
    """Generates a grid of coordinates within a bounding box."""
    print("\nGenerating a coordinate grid...")
    print(f"  Spacing: {spacing_meters} meters")
    min_easting, max_easting = (
        min(start_coords[0], end_coords[0]),
        max(start_coords[0], end_coords[0]),
    )
    min_northing, max_northing = (
        min(start_coords[1], end_coords[1]),
        max(start_coords[1], end_coords[1]),
    )

    easting_range = np.arange(min_easting, max_easting, spacing_meters)
    northing_range = np.arange(min_northing, max_northing, spacing_meters)

    if easting_range.size == 0 or northing_range.size == 0:
        print(
            "Warning: Grid is empty. Check coordinates and spacing. Returning empty DataFrame."
        )
        return pd.DataFrame({"easting": [], "northing": []})

    easting_grid, northing_grid = np.meshgrid(easting_range, northing_range)
    return pd.DataFrame(
        {"easting": easting_grid.flatten(), "northing": northing_grid.flatten()}
    )


def visualize_single_skyline(db_path: Path, point_id: str):
    """
    Loads and displays a single skyline mask from the NPZ database.
    """
    if not db_path.exists():
        print(f"Error: Database file not found at '{db_path}'")
        return

    print(f"\nVisualizing Skyline for Point ID: {point_id} ")

    with np.load(db_path) as data:
        if point_id not in data:
            print(f"Error: Point ID '{point_id}' not found in the database.")
            available_points = list(data.keys())
            print(
                f"Available points ({len(available_points)} total): {available_points[:5]}..."
            )
            return

        mask = data[point_id]

        plt.figure(figsize=(12, 3))
        plt.imshow(mask, cmap="gray", aspect="auto")
        plt.title(f"Skyline Mask for {point_id}\n(White = Sky, Black = Terrain)")
        plt.xlabel("Azimuth (Panorama Width)")
        plt.ylabel("Elevation (Panorama Height)")
        plt.show()


def generate_skyline_for_single_point(
    viewpoint_coords: Tuple[float, float, float],
) -> np.ndarray:
    """
    Generates a single skyline mask for a specific coordinate pair on-the-fly.
    This encapsulates the logic from the main pipeline's Stage 12 for one point.
    """
    point_id = f"validation_vp_{viewpoint_coords[0]:.0f}_{viewpoint_coords[1]:.0f}"
    print(f"\nGenerating DEM skyline for viewpoint: {point_id}...")

    all_laz_files = find_laz_files()

    # 1. Find relevant LAZ files and create tasks
    tasks = []
    for laz_file in all_laz_files:
        if check_file_relevance_from_json(laz_file, viewpoint_coords):
            tasks.append((point_id, viewpoint_coords, laz_file))

    if not tasks:
        print("No relevant LAZ files found for this coordinate.")
        return np.zeros(
            (constants.PANORAMA_IMG_HEIGHT, constants.PANORAMA_IMG_WIDTH),
            dtype=np.uint8,
        )

    print(f"Found {len(tasks)} relevant LAZ files to process for this point.")

    # 2. Run workers in parallel
    worker_config = {
        "SEARCH_RADIUS": constants.SEARCH_RADIUS,
        "ANGLE_STEP": constants.ANGLE_STEP,
        "MIN_DISTANCE": constants.MIN_DISTANCE,
    }
    num_processes = max(1, cpu_count() - 2)
    partial_results = []
    with Pool(
        processes=num_processes, initializer=init_worker, initargs=(worker_config,)
    ) as pool:
        iterator = pool.imap_unordered(process_viewpoint_and_file, tasks)
        for result in tqdm(iterator, total=len(tasks), desc=f"Processing"):
            if result and result[1] is not None:
                partial_results.append(result)

    # 3. Aggregate results
    num_bins = int(360 / constants.ANGLE_STEP)
    aggregated_elevations = np.full(num_bins, 0.0)
    for _, (azimuths, elevations) in partial_results:
        df = pd.DataFrame({"azimuth": azimuths, "elevation": elevations})
        angle_labels = np.arange(num_bins)
        angle_bin_edges = np.linspace(0, 360, num_bins + 1)
        df["bin"] = pd.cut(
            df["azimuth"],
            bins=angle_bin_edges,
            labels=angle_labels,
            right=False,
            include_lowest=True,
        )
        max_in_bin = df.groupby("bin", observed=True)["elevation"].max()
        new_maxes = np.full(num_bins, 0.0)
        new_maxes[max_in_bin.index] = max_in_bin.values
        aggregated_elevations = np.maximum(aggregated_elevations, new_maxes)

    # 4. Generate final mask array
    angles = np.linspace(0, 360, num_bins, endpoint=False)
    final_mask = generate_skyline_mask_array(
        angles,
        aggregated_elevations,
        constants.PANORAMA_IMG_WIDTH,
        constants.PANORAMA_IMG_HEIGHT,
    )

    print("Single DEM skyline generated.")
    return final_mask


def visualize_comparison(
    dem_mask: np.ndarray, real_mask: np.ndarray, point_id: str, real_image_name: str
):
    """
    Saves a side-by-side comparison of the generated DEM mask and the real segmented mask to a file.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle(f"Skyline Comparison for Viewpoint: {point_id}", fontsize=16)

    ax1.imshow(dem_mask, cmap="gray", aspect="auto")
    ax1.set_title("Generated DEM Skyline")
    ax1.set_ylabel("Elevation")
    ax1.set_xticks([])

    ax2.imshow(real_mask, cmap="gray", aspect="auto")
    ax2.set_title(f"Real Segmented Panorama ({real_image_name})")
    ax2.set_xlabel("Azimuth")
    ax2.set_ylabel("Elevation")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = Path("./output_plots")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"comparison_{point_id}.png"

    plt.savefig(save_path)
    plt.close(fig)

    # Telling the user where to find the image
    print(f"Comparison plot saved to: {save_path}")
