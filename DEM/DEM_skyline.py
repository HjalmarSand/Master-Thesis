import json
import sys
from pathlib import Path
from typing import List, Tuple

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
from rtree import index
from scipy.signal import medfilt
from tqdm import tqdm

import constants

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# This global variable will be initialized in each worker process
global_constants = {}


def create_lidar_chunks(
    source_dir: Path,
    chunk_dir: Path,
    manifest_path: Path,
    chunk_tile_size: int = 500,  # Changed from capacity to tile size in meters
):
    """
    Splits large LAZ files into smaller, square chunks (tiles) and creates a
    manifest file with their spatial bounds. Skips files already in the manifest.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    source_files = find_laz_files(source_dir)

    all_chunk_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            all_chunk_data = json.load(f)
        print(f"Loaded {len(all_chunk_data)} existing chunk records from manifest.")

    processed_stems = {key.split("_chunk_")[0] for key in all_chunk_data.keys()}

    for laz_file in tqdm(source_files, desc="Chunking LAZ Files"):
        file_id = laz_file.stem
        if file_id in processed_stems:
            continue

        print(f"\nProcessing and chunking '{laz_file.name}'...")
        # Using a slightly different output template for tiled chunks
        output_template = chunk_dir / f"{file_id}_chunk_#.laz"
        pipeline_json = {
            "pipeline": [
                str(laz_file),
                {
                    "type": "filters.splitter",
                    "length": chunk_tile_size,  # FIXED: Use 'length' for spatial tiling
                },
                {
                    "type": "writers.las",
                    "filename": str(output_template),
                },
            ]
        }
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        # Update glob pattern to match the new tiled output format
        created_chunks = sorted(chunk_dir.glob(f"{file_id}_chunk_*.laz"))
        for chunk_path in created_chunks:
            try:
                with laspy.open(chunk_path) as chunk_file:
                    header = chunk_file.header
                    chunk_bounds = {
                        "min_x": header.x_min,
                        "max_x": header.x_max,
                        "min_y": header.y_min,
                        "max_y": header.y_max,
                    }
                all_chunk_data[chunk_path.stem] = {
                    "path": str(chunk_path),
                    "bounds": chunk_bounds,
                }
            except Exception as e:
                print(f"Warning: Could not read header for {chunk_path.name}: {e}")

    with open(manifest_path, "w") as f:
        json.dump(all_chunk_data, f, indent=2)
    print(f"Chunking manifest updated. Total chunks: {len(all_chunk_data)}")


def find_laz_files(directory: Path) -> List[Path]:
    """
    Find all LAZ files in the specified directory.
    """
    print(f"Searching for LAZ files in directory: {directory}")
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    laz_files = sorted(list(directory.glob("*.laz")) + list(directory.glob("*.LAZ")))
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in directory: {directory}")
    print(f"Found {len(laz_files)} LAZ files.")
    return laz_files


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


def calculate_skyline_similarity(dem_mask: np.ndarray, real_mask: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) score between two skyline masks.
    """
    # Convert masks to boolean (True for sky, False for terrain)
    dem_sky = dem_mask > 0
    real_sky = real_mask > 0

    intersection = np.sum(np.logical_and(dem_sky, real_sky))
    union = np.sum(np.logical_or(dem_sky, real_sky))

    return intersection / union if union > 0 else 0.0


def visualize_comparison(
    original_real_mask: np.ndarray,
    dem_mask: np.ndarray,
    aligned_real_mask: np.ndarray,
    point_id: str,
    real_image_name: str,
    similarity_score: float,
    yaw_offset: float,
    save_path: Path,
):
    """
    Saves a 3-panel plot comparing the original, DEM, and aligned real skylines.
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 9))
    title = (
        f"Comparison for Query '{real_image_name}' (Point: {point_id})\n"
        f"Calculated Yaw Offset: {yaw_offset:.2f}° | "
        f"Post-Alignment IoU Score: {similarity_score:.2%}"
    )
    fig.suptitle(title, fontsize=14)

    # Plot 1: Original Real Panorama
    axs[0].imshow(original_real_mask, cmap="gray", aspect="auto")
    axs[0].set_title("1. Original Real Panorama (Unaligned)")
    axs[0].set_ylabel("Elevation")

    # Plot 2: DEM Skyline (Correctly Oriented)
    axs[1].imshow(dem_mask, cmap="gray", aspect="auto")
    axs[1].set_title("2. DEM Skyline (North-Aligned)")
    axs[1].set_ylabel("Elevation")

    # Plot 3: Aligned Real Panorama
    axs[2].imshow(aligned_real_mask, cmap="gray", aspect="auto")
    axs[2].set_title("3. Real Panorama (Aligned to DEM)")
    axs[2].set_xlabel("Azimuth")
    axs[2].set_ylabel("Elevation")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def process_viewpoint_against_chunks(task: tuple) -> tuple[str, np.ndarray | None]:
    """
    Worker function to generate a complete skyline for a SINGLE viewpoint by
    iterating through all relevant LiDAR chunks. This is the memory-safe worker.
    """
    point_id, viewpoint_coords, chunk_paths = task
    vp_x, vp_y, vp_z = viewpoint_coords

    num_bins = int(360 / constants.ANGLE_STEP)
    min_distance_sq = constants.MIN_DISTANCE**2

    # We still initialize with -999.0 to correctly calculate maximums,
    # especially for points with negative elevation (e.g., water).
    final_skyline = np.full(num_bins, -999.0, dtype=np.float32)

    try:
        for chunk_path in chunk_paths:
            pipeline_json = f'["{str(chunk_path)}"]'
            pipeline = pdal.Pipeline(pipeline_json)
            if pipeline.execute() == 0:
                continue

            points = pipeline.arrays[0]
            if len(points) == 0:
                continue

            rel_x = points["X"] - vp_x
            rel_y = points["Y"] - vp_y
            horizontal_dist_sq = rel_x**2 + rel_y**2

            valid_mask = horizontal_dist_sq > min_distance_sq
            if not np.any(valid_mask):
                continue

            valid_rel_x = rel_x[valid_mask]
            valid_rel_y = rel_y[valid_mask]
            valid_rel_z = points["Z"][valid_mask] - vp_z
            valid_dist = np.sqrt(horizontal_dist_sq[valid_mask])

            azimuth_deg = (np.degrees(np.arctan2(valid_rel_x, valid_rel_y)) + 360) % 360
            elevation_deg = np.degrees(np.arctan2(valid_rel_z, valid_dist))

            angle_bin_edges = np.linspace(0, 360, num_bins + 1)
            bins = np.digitize(azimuth_deg, bins=angle_bin_edges) - 1
            bins = np.clip(bins, 0, num_bins - 1)

            np.maximum.at(final_skyline, bins, elevation_deg)

        if np.any(final_skyline > -999.0):
            # *** NEW LINE: Before returning, set all remaining -999.0 values to 0.0 ***
            final_skyline[final_skyline == -999.0] = 0.0
            return (point_id, final_skyline)
        else:
            return (point_id, None)

    except Exception:
        return (point_id, None)


def create_simplified_chunks(
    raw_chunk_dir: Path, simplified_chunk_dir: Path, pipeline_template_path: Path
):
    """
    Applies a PDAL pipeline to each raw chunk to create a simplified version.
    Includes resume logic to skip already processed chunks.
    """
    simplified_chunk_dir.mkdir(parents=True, exist_ok=True)

    with open(pipeline_template_path, "r") as f:
        pipeline_template = json.load(f)

    raw_chunks = sorted(list(raw_chunk_dir.glob("*.laz")))
    if not raw_chunks:
        print(f"Warning: No raw chunks found in {raw_chunk_dir}")
        return

    print(f"Found {len(raw_chunks)} raw chunks to process.")

    for raw_chunk_path in tqdm(raw_chunks, desc="Simplifying Chunks"):
        simplified_chunk_path = simplified_chunk_dir / raw_chunk_path.name

        # Resume logic: skip if the output file already exists
        if simplified_chunk_path.exists():
            continue

        # Modify the pipeline template with the correct file paths
        pipeline_template[0]["filename"] = str(raw_chunk_path)
        pipeline_template[-1]["filename"] = str(simplified_chunk_path)

        try:
            pipeline = pdal.Pipeline(json.dumps(pipeline_template))
            pipeline.execute()
        except Exception as e:
            print(f"Error processing {raw_chunk_path.name}: {e}")


def generate_skyline_for_single_point(
    viewpoint_coords: Tuple[float, float, float],
    simplified_manifest_path: Path,
) -> np.ndarray | None:
    """
    Generates a single skyline mask for a specific coordinate pair on-the-fly
    using the modern, efficient chunk-based method.
    """
    point_id = f"on_the_fly_vp_{viewpoint_coords[0]:.0f}_{viewpoint_coords[1]:.0f}"
    print(f"\nGenerating DEM skyline for viewpoint: {point_id}...")

    if not simplified_manifest_path.exists():
        print(
            f"ERROR: Simplified chunk manifest not found at '{simplified_manifest_path}'"
        )
        print("Please run Stage 10 to prepare LiDAR assets first.")
        return None

    # 1. Load manifest and build R-tree index
    with open(simplified_manifest_path, "r") as f:
        chunk_manifest = json.load(f)

    p = index.Property(dimension=2)
    idx = index.Index(properties=p)
    chunk_paths_map = {}
    for i, (stem, data) in enumerate(chunk_manifest.items()):
        b = data["bounds"]
        idx.insert(i, (b["min_x"], b["min_y"], b["max_x"], b["max_y"]))
        chunk_paths_map[i] = Path(data["path"])

    # 2. Find relevant chunks for the viewpoint
    vp_x, vp_y, _ = viewpoint_coords
    search_radius = constants.SEARCH_RADIUS
    search_bounds = (
        vp_x - search_radius,
        vp_y - search_radius,
        vp_x + search_radius,
        vp_y + search_radius,
    )
    relevant_indices = list(idx.intersection(search_bounds))

    if not relevant_indices:
        print("No relevant LiDAR chunks found for this coordinate.")
        return None

    relevant_paths = [chunk_paths_map[i] for i in relevant_indices]
    print(f"Found {len(relevant_paths)} relevant chunks to process.")

    # 3. Create and run the task
    # Note: we pass the constants directly since we are not using a multiprocessing Pool here
    task = (point_id, viewpoint_coords, relevant_paths)

    # We process this single point directly
    skyline_result = process_viewpoint_against_chunks(task)

    if skyline_result is None or skyline_result[1] is None:
        print("Skyline generation failed for this point (no valid points found).")
        return None

    _, skyline_vector = skyline_result

    # 4. Generate final mask array
    num_bins = len(skyline_vector)
    angles = np.linspace(0, 360, num_bins, endpoint=False)
    final_mask = generate_skyline_mask_array(
        angles,
        skyline_vector,
        constants.PANORAMA_IMG_WIDTH,
        constants.PANORAMA_IMG_HEIGHT,
    )

    print("Single DEM skyline generated successfully.")
    return final_mask
