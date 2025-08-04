import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

# --- Standard Library Imports ---
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
from scipy.signal import medfilt
from tqdm import tqdm


# --- Configuration Section ---
# This mock class replaces your 'constants.py' for this standalone script.
# IMPORTANT: Adjust these values to match your project's 'constants.py' file.
class MockConstants:
    LIDAR_DATA_DIR = "./lidar_data"  # Path to your folder of .laz files
    SEARCH_RADIUS = 2000  # (meters) How far to look for points from the viewpoint
    MIN_DISTANCE = 5  # (meters) Ignore points closer than this to the viewpoint
    ANGLE_STEP = 0.5  # (degrees) Resolution for azimuth bins
    PANORAMA_IMG_WIDTH = 1024
    PANORAMA_IMG_HEIGHT = 512
    CAMERA_HEIGHT_ABOVE_GROUND = 1.6  # (meters) Assumed height of camera


constants = MockConstants()


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) for two binary masks."""
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    return intersection / union if union > 0 else 0.0


def find_best_shift(
    dem_mask: np.ndarray, cnn_mask: np.ndarray, plot_scores=False
) -> Tuple[int, float]:
    """Finds the best horizontal circular shift to align cnn_mask to dem_mask."""
    if dem_mask.shape != cnn_mask.shape:
        print(
            "Warning: Masks have different sizes. Resizing CNN mask to match DEM mask."
        )
        cnn_mask = cv2.resize(
            cnn_mask,
            (dem_mask.shape[1], dem_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    dem_mask_binary = dem_mask > 0
    cnn_mask_binary = cnn_mask > 0

    _, width = dem_mask.shape
    best_shift = 0
    max_iou = 0
    scores = []

    for shift in range(width):
        shifted_panorama = np.roll(cnn_mask_binary, shift, axis=1)
        iou = calculate_iou(dem_mask_binary, shifted_panorama)
        scores.append(iou)
        if iou > max_iou:
            max_iou = iou
            best_shift = shift

    if plot_scores:
        plt.figure(figsize=(10, 4))
        plt.plot(range(width), scores)
        plt.title("IoU Score vs. Horizontal Shift")
        plt.xlabel("Shift (pixels)")
        plt.ylabel("IoU Score")
        plt.axvline(
            x=best_shift, color="r", linestyle="--", label=f"Best Shift: {best_shift}px"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_shift, max_iou


def apply_shift(image: np.ndarray, shift: int) -> np.ndarray:
    """Applies the calculated circular shift to an image."""
    return np.roll(image, shift, axis=1)


def find_laz_files() -> List[Path]:
    """Find all LAZ files in the configured directory."""
    laz_directory = Path(constants.LIDAR_DATA_DIR)
    if not laz_directory.exists():
        raise FileNotFoundError(f"LIDAR data directory not found: {laz_directory}")
    laz_files = sorted(
        list(laz_directory.glob("*.laz")) + list(laz_directory.glob("*.LAZ"))
    )
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in directory: {laz_directory}")
    return laz_files


def get_ground_elevation(
    easting: float, northing: float, all_laz_files: List[Path]
) -> float:
    """
    Estimates ground elevation (Z) at a given (Easting, Northing) coordinate.
    It finds the closest LAZ file and queries the Z value of the lowest point in a small area.
    """
    # For simplicity, we'll just use the first LAZ file. A more robust solution
    # would be to find the file whose bounds contain the point.
    # This is a reasonable approximation if all your LAZ files are for the same area.
    target_file = all_laz_files[0]

    # Define a small box around the point to find the ground
    box_size = 10  # 10m x 10m box
    min_x, max_x = easting - box_size / 2, easting + box_size / 2
    min_y, max_y = northing - box_size / 2, northing + box_size / 2

    pipeline_json = {
        "pipeline": [
            {"type": "readers.las", "filename": str(target_file)},
            {
                "type": "filters.crop",
                "bounds": f"([{min_x}, {max_x}],[{min_y}, {max_y}])",
            },
            {"type": "filters.sort", "dimension": "Z", "order": "ASC"},
            {"type": "filters.head", "count": 1},
        ]
    }
    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        if pipeline.execute() > 0:
            points = pipeline.arrays[0]
            if len(points) > 0:
                return points["Z"][0]
    except Exception as e:
        print(
            f"Warning: Could not determine ground elevation for ({easting}, {northing}). Error: {e}"
        )

    print(
        "Warning: Defaulting ground elevation to 0. This may affect skyline accuracy."
    )
    return 0.0


def generate_skyline_mask_array(
    angles: np.ndarray, elevations: np.ndarray
) -> np.ndarray:
    """Creates a binary mask array from aggregated skyline data."""
    output_width = constants.PANORAMA_IMG_WIDTH
    output_height = constants.PANORAMA_IMG_HEIGHT

    sort_indices = np.argsort(angles)
    sorted_angles, sorted_elevations = angles[sort_indices], elevations[sort_indices]

    wrapped_angles = np.concatenate(
        ([sorted_angles[-1] - 360], sorted_angles, [sorted_angles[0] + 360])
    )
    wrapped_elevations = np.concatenate(
        ([sorted_elevations[-1]], sorted_elevations, [sorted_elevations[0]])
    )

    output_azimuths = np.linspace(0, 360, output_width, endpoint=False)
    interpolated_elevations = np.interp(
        output_azimuths, wrapped_angles, wrapped_elevations
    )

    filtered_elevations = medfilt(interpolated_elevations, kernel_size=19)

    # THIS IS THE CRUCIAL PART: Your original code correctly centers North.
    # It calculates elevations based on a 0-360 degree system where 0 is North.
    # It then rolls the array so that 0 degrees (North) is at the center of the image.
    pixels_to_roll = -int(output_width / 2)
    final_elevations = np.roll(filtered_elevations, pixels_to_roll)

    v_fov = 90.0
    top_elevation = v_fov / 2.0
    skyline_rows = np.clip(
        np.round(((top_elevation - final_elevations) / v_fov) * output_height),
        0,
        output_height,
    ).astype(int)

    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    for x, y_sky in enumerate(skyline_rows):
        mask[0:y_sky, x] = 255
    return mask


# This global variable will be initialized in each worker process
global_worker_config = {}


def init_worker(config: dict):
    """Initializes each worker process with shared configuration."""
    global global_worker_config
    global_worker_config = config


def process_viewpoint_and_file(task: Tuple) -> Tuple:
    """Worker function: processes ONE LAZ file for ONE viewpoint."""
    point_id, viewpoint_coords, file_path = task
    try:
        vp_x, vp_y, vp_z = viewpoint_coords
        min_x, max_x = vp_x - constants.SEARCH_RADIUS, vp_x + constants.SEARCH_RADIUS
        min_y, max_y = vp_y - constants.SEARCH_RADIUS, vp_y + constants.SEARCH_RADIUS

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
            return (point_id, None)

        points = pipeline.arrays[0]
        if len(points) == 0:
            return (point_id, None)

        rel_x = points["X"] - vp_x
        rel_y = points["Y"] - vp_y
        rel_z = points["Z"] - vp_z

        horizontal_dist = np.sqrt(rel_x**2 + rel_y**2)
        valid_mask = horizontal_dist > constants.MIN_DISTANCE
        if not np.any(valid_mask):
            return (point_id, None)

        # Azimuth: 0 is North (Y-axis), 90 is East (X-axis)
        azimuth_deg = (
            np.degrees(np.arctan2(rel_x[valid_mask], rel_y[valid_mask])) + 360
        ) % 360
        elevation_deg = np.degrees(
            np.arctan2(rel_z[valid_mock], horizontal_dist[valid_mask])
        )

        return (point_id, (azimuth_deg, elevation_deg))
    except Exception:
        return (point_id, None)


def generate_dem_skyline(
    viewpoint_coords: Tuple[float, float, float], all_laz_files: List[Path]
) -> np.ndarray:
    """Generates a single DEM skyline mask for a specific coordinate."""
    point_id = f"vp_{viewpoint_coords[0]:.0f}_{viewpoint_coords[1]:.0f}"

    tasks = [(point_id, viewpoint_coords, laz_file) for laz_file in all_laz_files]
    if not tasks:
        print("No LAZ files found to process.")
        return np.zeros(
            (constants.PANORAMA_IMG_HEIGHT, constants.PANORAMA_IMG_WIDTH),
            dtype=np.uint8,
        )

    num_processes = max(1, cpu_count() - 2)
    partial_results = []

    with Pool(processes=num_processes) as pool:
        iterator = pool.imap_unordered(process_viewpoint_and_file, tasks)
        for result in tqdm(
            iterator,
            total=len(tasks),
            desc=f"  Generating DEM for {point_id}",
            leave=False,
        ):
            if result and result[1] is not None:
                partial_results.append(result)

    if not partial_results:
        print(
            f"Warning: No points found within search radius for {point_id}. Returning empty mask."
        )
        return np.zeros(
            (constants.PANORAMA_IMG_HEIGHT, constants.PANORAMA_IMG_WIDTH),
            dtype=np.uint8,
        )

    num_bins = int(360 / constants.ANGLE_STEP)
    aggregated_elevations = np.full(
        num_bins, -90.0
    )  # Initialize with lowest possible elevation

    all_azimuths = np.concatenate([res[1][0] for res in partial_results])
    all_elevations = np.concatenate([res[1][1] for res in partial_results])

    # Binning to find max elevation per azimuth slice
    bins = np.linspace(0, 360, num_bins + 1)
    indices = np.digitize(all_azimuths, bins) - 1

    for i in range(num_bins):
        mask = indices == i
        if np.any(mask):
            aggregated_elevations[i] = np.max(all_elevations[mask])

    angles = np.linspace(0, 360, num_bins, endpoint=False)
    final_mask = generate_skyline_mask_array(angles, aggregated_elevations)
    return final_mask


def main():
    """
    Main function to drive the alignment process.
    Reads a CSV, and for each row, generates a DEM skyline, loads a CNN mask,
    aligns them, and saves the result.
    """

    input_csv_path = Path("viewpoints.csv")
    cnn_masks_dir = Path("cnn_masks")
    output_dir = Path("output_aligned_masks")
    output_dir.mkdir(exist_ok=True)

    # --- Load Data ---
    if not input_csv_path.exists():
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return

    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} viewpoints from '{input_csv_path}'.")

    print("Finding all LAZ files...")
    all_laz_files = find_laz_files()
    print(f"Found {len(all_laz_files)} LAZ files in '{constants.LIDAR_DATA_DIR}'.")

    # --- Process Each Viewpoint ---
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing All Viewpoints"
    ):
        try:
            easting = row["easting"]
            northing = row["northing"]
            cnn_mask_filename = row["filename"]
            cnn_mask_path = cnn_masks_dir / cnn_mask_filename

            print(f"\nProcessing viewpoint {index + 1}/{len(df)}: {cnn_mask_filename}")

            if not cnn_mask_path.exists():
                print(
                    f"  -> ‼️ Error: CNN mask file not found: '{cnn_mask_path}'. Skipping."
                )
                continue

            # 1. Generate DEM Skyline
            print(
                f"  -> Estimating ground elevation at ({easting:.2f}, {northing:.2f})..."
            )
            ground_z = get_ground_elevation(easting, northing, all_laz_files)
            viewpoint_coords = (
                easting,
                northing,
                ground_z + constants.CAMERA_HEIGHT_ABOVE_GROUND,
            )

            dem_skyline_mask = generate_dem_skyline(viewpoint_coords, all_laz_files)

            # 2. Load CNN Panorama Mask
            cnn_mask = cv2.imread(str(cnn_mask_path), cv2.IMREAD_GRAYSCALE)
            if cnn_mask is None:
                print(
                    f"  -> ‼️ Error: Could not load CNN mask '{cnn_mask_path}'. Skipping."
                )
                continue

            # 3. Find Best Shift to Align
            print("  -> Calculating optimal alignment shift...")
            best_shift, iou_score = find_best_shift(dem_skyline_mask, cnn_mask)
            print(
                f"  -> Best shift: {best_shift} pixels with IoU score: {iou_score:.4f}"
            )

            # 4. Apply Shift and Save
            aligned_cnn_mask = apply_shift(cnn_mask, best_shift)

            output_filename = f"aligned_{cnn_mask_filename}"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), aligned_cnn_mask)
            print(f"  -> ✅ Saved aligned mask to '{output_path}'")

        except Exception as e:
            print(
                f"  -> ‼️ An unexpected error occurred for row {index}: {e}. Skipping."
            )

    print("\n--- Alignment process complete! ---")


if __name__ == "__main__":
    main()
