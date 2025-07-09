import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import constants
from image_retrieval.colmap_naive_image_retrieval import (
    ColmapImageRetriever,
    retrieve_top_matches,
)
from utils import (
    fourier_skyline_encoder,
    fourier_skyline_matcher,
    orientation_path_from_gps,
    split_stacked_images,
)

# from utils.use_polarmist_undistorter import main as fisheye_undistorter_main
from utils.create_panoramas import build_panoramas
from utils.DEM_skyline import create_and_save_grid, create_skyline_database

# Use only one of these two, either polar mist undistortion from their estimated campera parameters our ours from the given csv from camera manifacturer
from utils.fisheye_undistorter import FisheyeUndistorter

# from utils.skyline_segmenter import segment_skyline_from_panoramas, get_sam_predictor
from utils.segformer_segmenter import SemanticSegmenter
from utils.utility import get_original_numbered_file_name


def print_separator(title=""):
    """Print a visual separator for pipeline stages."""
    print("\n" + "=" * 60)
    if title:
        print(f" {title} ")
        print("=" * 60)
    else:
        print("=" * 60)


def print_stage_header(stage_num, stage_name, description=""):
    """Print a formatted header for each pipeline stage."""
    print(f"\nSTAGE {stage_num}: {stage_name}")
    if description:
        print(f"   {description}")
    print("-" * 50)


def get_undistortion_parameters(dataset_dir: Path) -> dict:
    """
    Returns a dictionary of undistortion parameters based on the dataset name.
    """
    dataset_name = dataset_dir.name
    print(f"Getting parameters for '{dataset_name}'...")

    if dataset_name == "dataset1":
        # Parameters specific to dataset1
        return {
            "vertical_offset": 17,
            "horizon_bend_cam1": 15,
            "horizon_bend_cam2": -15,
            "rotation_cam1": 9.44,
            "rotation_cam2": -9.85,
        }
    else:
        # Default parameters for all other datasets
        return {
            "vertical_offset": 17,
            "horizon_bend_cam1": 15,
            "horizon_bend_cam2": -15,
            "rotation_cam1": 5.93,
            "rotation_cam2": -6.48,
        }


def run_image_splitting():
    """
    Stage 1: Split stacked images into individual camera images.
    """
    print_stage_header(
        1, "IMAGE SPLITTING", "Splitting stacked images into cam1 and cam2 images"
    )

    try:
        # Define the path to the folder containing the original dataset folders
        datasets_root = Path(constants.IMAGES_DIR) / "original_datasets"

        exit_code = split_stacked_images.run_splitter(datasets_root)

        if exit_code == 0 or exit_code is None:
            print("Image splitting completed successfully!")
            return True
        else:
            print("Image splitting failed!")
            return False

    except Exception as e:
        print(f"Error during image splitting: {e}")
        return False


# TODO: We should probably move stage 3 to here, and then make the undistortion stage 3 instead. The reason
# is that right now we do not have any normalization accounting for the rocking of the boat. We could perhaps
# use the projectory that we find between coordinates in the gps csv to predict the sway of the boat and account
# for that in a dynamic roll. This is functionality that needs to be added to fisheye undistorter, where we
# poll the csv before creating each image to find a roll. Probably we want to view the curve between the gps points
# as Bezier curves.
#
# Perhaps we can come up with a way to identify the angle of horizon dynamically in each image and account for a roll from that
# but that does not seem like a time worthy approach as we would need to accurately annotate probably thousands of images
# and then train a neural network, hoping for good results in a timely manner, with no guarantees. Tuning estimated values from
# Bezier curves is likely a better use of our time.


def run_fisheye_undistortion():
    """
    Stage 2: Undistort fisheye images for all datasets.
    - Finds each dataset folder in the images directory.
    - Applies roll and horizon bend correction and a symmetrical vertical pitch correction to both cameras.
    """
    print_stage_header(
        2,
        "FISHEYE UNDISTORTION",
        "Applying camera-specific alignment parameters to a selected dataset.",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find and sort datasets that have been processed by Stage 1
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "distorted_images_cam1").is_dir()
            ]
        )

        if not dataset_dirs:
            print(
                "No processed dataset folders found (e.g., 'images/dataset1/distorted_images_cam1')."
            )
            print("Please run Stage 1 first.")
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to process:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")

            choice_str = input("Enter the number of the dataset: ")

            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print(
                        f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(f"\n--- Processing dataset: {selected_dir.name} ---")

        # Initialize the undistorter
        undistorter = FisheyeUndistorter(csv_path=constants.DISTORTION_PARAMS_CSV)

        # Get the specific parameters for this dataset
        params = get_undistortion_parameters(selected_dir)

        # Calculate pitch adjustment based on the retrieved parameters
        half_offset = params["vertical_offset"] / 2.0
        pitch_adjustment_angle = (
            FisheyeUndistorter.calculate_pitch_correction_from_offset(
                half_offset, constants.OUTPUT_IMG_WIDTH, 90
            )
        )
        print(f"Applying pitch adjustment of +/- {pitch_adjustment_angle:.4f} degrees.")

        # Dynamically define paths for the selected dataset
        input_dir_cam1 = selected_dir / "distorted_images_cam1"
        output_dir_cam1_left = selected_dir / "undistorted_cam1_left"
        output_dir_cam1_right = selected_dir / "undistorted_cam1_right"

        input_dir_cam2 = selected_dir / "distorted_images_cam2"
        output_dir_cam2_left = selected_dir / "undistorted_cam2_left"
        output_dir_cam2_right = selected_dir / "undistorted_cam2_right"

        # Process Cam1
        print(f"\nProcessing Camera 1 for {selected_dir.name} (pitching down)...")
        undistorter.batch_process_images(
            input_dir=input_dir_cam1,
            output_dir_left=output_dir_cam1_left,
            output_dir_right=output_dir_cam1_right,
            output_width=constants.OUTPUT_IMG_WIDTH,
            output_height=constants.OUTPUT_IMG_HEIGHT,
            rectilinear_fov_degrees=90,
            horizontal_angle_offset_degrees=45,
            vertical_angle_offset_degrees=-pitch_adjustment_angle,
            rotation_degrees=params["rotation_cam1"],
            horizon_bend_correction=params["horizon_bend_cam1"],
        )

        # Process Cam2
        print(f"\nProcessing Camera 2 for {selected_dir.name} (pitching up)...")
        undistorter.batch_process_images(
            input_dir=input_dir_cam2,
            output_dir_left=output_dir_cam2_left,
            output_dir_right=output_dir_cam2_right,
            output_width=constants.OUTPUT_IMG_WIDTH,
            output_height=constants.OUTPUT_IMG_HEIGHT,
            rectilinear_fov_degrees=90,
            horizontal_angle_offset_degrees=45,
            vertical_angle_offset_degrees=pitch_adjustment_angle,
            rotation_degrees=params["rotation_cam2"],
            horizon_bend_correction=params["horizon_bend_cam2"],
        )

        print(
            f"\n✅ Fisheye undistortion for '{selected_dir.name}' finished successfully!"
        )
        return True

    except Exception as e:
        print(f"An error occurred during fisheye undistortion: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_gps_orientation_processing():
    """
    Stage 3: Interactively select a dataset and process its GPS data to calculate orientation.
    """
    print_stage_header(
        3,
        "GPS & ORIENTATION PROCESSING",
        "Calculates orientation from GPS data for a selected dataset.",
    )

    try:
        INPUT_CSV_NAME = "gps_interpolated.csv"
        source_datasets_dir = Path(constants.IMAGES_DIR) / "original_datasets"

        available_datasets = sorted(
            [
                d
                for d in source_datasets_dir.iterdir()
                if d.is_dir() and (d / INPUT_CSV_NAME).is_file()
            ]
        )

        if not available_datasets:
            print(
                f"No source dataset folders found containing '{INPUT_CSV_NAME}'. Searched in: {source_datasets_dir}"
            )
            return False

        selected_source_dir = None
        while True:
            print("\nPlease select a source dataset to process for GPS orientation:")
            for i, dir_path in enumerate(available_datasets):
                print(f"  [{i}]: {dir_path.name}")

            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_datasets):
                    selected_source_dir = available_datasets[choice_idx]
                    break
                else:
                    print(
                        f"\n❌ Invalid input. Please enter a number between 0 and {len(available_datasets) - 1}."
                    )
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(f"\n--- Processing GPS data for: {selected_source_dir.name} ---")

        input_csv_path = selected_source_dir / INPUT_CSV_NAME
        output_dir = Path(constants.IMAGES_DIR) / selected_source_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_csv_path = output_dir / f"{input_csv_path.stem}_with_orientation.csv"
        output_plot_path = output_dir / f"{input_csv_path.stem}_trajectory.png"

        print(f"Reading GPS data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        print("Calculating orientation...")
        df["Orientation"] = orientation_path_from_gps.calculate_orientation(df)

        print(f"Saving data with orientation to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False, float_format="%.3f")

        orientation_path_from_gps.plot_trajectory(
            df,
            title_suffix=f"from {selected_source_dir.name}",
            save_path=output_plot_path,
        )

        print(
            f"\n✅ GPS orientation processing for '{selected_source_dir.name}' complete."
        )
        return True

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find file: {e}")
        return False
    except Exception as e:
        print(f"\n❌ An error occurred during GPS orientation processing: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_panorama_stitching():
    """
    Stage 4: Interactively select a dataset and stitch its four rectilinear views into panoramas.
    """
    print_stage_header(
        4,
        "PANORAMA STITCHING",
        "Combining four views into a single panoramic image per timestamp for a selected dataset",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find and sort datasets that have been processed by Stage 2
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "undistorted_cam1_left").is_dir()
            ]
        )

        if not dataset_dirs:
            print(
                "No undistorted dataset folders found (e.g., 'images/dataset1/undistorted_cam1_left')."
            )
            print("Please run Stage 2 first.")
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to stitch into panoramas:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")

            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print(
                        f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(f"\n--- Stitching panoramas for dataset: {selected_dir.name} ---")

        # Dynamically find the orientation CSV for the selected dataset
        orientation_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        if not orientation_csv_path.is_file():
            print(
                f"❌ Orientation CSV not found for {selected_dir.name} at '{orientation_csv_path}'."
            )
            print("   Please run Stage 3 for this dataset first.")
            return False

        # Define input and output paths for the selected dataset
        cam1_left_dir = selected_dir / "undistorted_cam1_left"
        cam1_right_dir = selected_dir / "undistorted_cam1_right"
        cam2_left_dir = selected_dir / "undistorted_cam2_left"
        cam2_right_dir = selected_dir / "undistorted_cam2_right"
        output_panorama_dir = selected_dir / "panoramas"

        if selected_dir == "dataset1":
            manual_rotation_offset_deg = 16  # TODO: Verify with better calculations
        else:
            manual_rotation_offset_deg = (
                16  # TODO: Change for the camera setup in dataset 2 and 3
            )

        exit_code = build_panoramas(
            cam1_left_dir=cam1_left_dir,
            cam1_right_dir=cam1_right_dir,
            cam2_left_dir=cam2_left_dir,
            cam2_right_dir=cam2_right_dir,
            output_dir=output_panorama_dir,
            orientation_csv_path=orientation_csv_path,
            manual_rotation_offset_deg=manual_rotation_offset_deg,
        )

        if exit_code == 0:
            print(
                f"\nPanorama stitching completed successfully for {selected_dir.name}!"
            )
            return True
        else:
            print(f"\nPanorama stitching failed for {selected_dir.name}!")
            return False

    except Exception as e:
        print(f"An error occurred during panorama stitching: {e}")
        import traceback

        traceback.print_exc()
        return False


# TODO: Right now we have only implemented for panoramas. We probably want to implement for fisheye images as well.
# TODO: Above comment is just the start of how some of these methods need to be adapted to make comparissons on fisheye as well
# TODO: For example, we might want to stitch together the fisheye images into panoramas as well to compare performance, or keep
# TODO: as is, both for skyline comparisson and classic image retrieval. Some of these methods also need to be updated for
# TODO: the new undistorted images that we have, with path changes and so on.
# def run_skyline_segmentation():
#    """
#    Stage 5: Use a prompted SAM Predictor to create binary skyline masks from panoramas.
#    """
#    print_stage_header(5, "SKYLINE SEGMENTATION",
#                    "Generating binary sky masks using a prompted SAM predictor")
#
#    try:
#        predictor = get_sam_predictor()
#        if predictor is None:
#            print("Could not initialize SAM predictor. Aborting stage.")
#            return False
#
#        exit_code = segment_skyline_from_panoramas(
#            panorama_dir=constants.PANORAMAS_DIR,
#            output_dir=constants.SKYLINE_MASKS_DIR,
#            predictor=predictor
#        )
#
#        if exit_code == 0:
#            print("Skyline segmentation completed successfully!")
#            return True
#        else:
#            print("Skyline segmentation failed!")
#            return False
#
#    except Exception as e:
#        print(f"An error occurred during skyline segmentation: {e}")
#        import traceback
#        traceback.print_exc()
#        return False


# This method instead of SAM uses segformer-b5
def run_skyline_segmentation():
    """
    Stage 5: Interactively select a dataset and use segformer-b5 to create
    binary skyline masks from its panoramas.
    """
    print_stage_header(
        5,
        "SKYLINE SEGMENTATION",
        "Generating binary sky masks using segformer-b5 for a selected dataset",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find and sort datasets that have been processed by Stage 4
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "panoramas").is_dir()
            ]
        )

        if not dataset_dirs:
            print("No panorama folders found (e.g., 'images/dataset1/panoramas').")
            print("Please run Stage 4 first.")
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to process:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")

            choice_str = input("Enter the number of the dataset: ")

            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break  # Exit loop on valid choice
                else:
                    print(
                        f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(
            f"\n--- Segmenting skylines for selected dataset: {selected_dir.name} ---"
        )

        segmenter = SemanticSegmenter()
        if segmenter.model is None:
            print("Could not initialize SemanticSegmenter model.")
            return False

        input_panorama_dir = selected_dir / "panoramas"
        output_binary_dir = selected_dir / "binary_panorama"
        output_overlay_dir = selected_dir / "overlay_binary_panorama"

        exit_code = segmenter.process_directory(
            input_dir=input_panorama_dir,
            output_dir_black_white=output_binary_dir,
            output_dir_overlay=output_overlay_dir,
        )

        if exit_code == 0:
            print(
                f"\n✅ Skyline segmentation completed successfully for {selected_dir.name}!"
            )
            return True
        else:
            print(f"\n❌ Skyline segmentation failed for {selected_dir.name}!")
            return False

    except Exception as e:
        print(f"An error occurred during skyline segmentation: {e}")
        import traceback

        traceback.print_exc()
        return False


def build_colmap_databases(force_rebuild: bool = False):
    """
    Stage 6: Build COLMAP databases for image retrieval for each dataset.
    """
    print_stage_header(
        6, "BUILD COLMAP DATABASES", "Building feature databases for each dataset"
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = [
            d
            for d in datasets_root_dir.iterdir()
            if d.is_dir() and (d / "distorted_images_cam1").is_dir()
        ]

        if not dataset_dirs:
            print(
                "No dataset folders with distorted images found. Please run Stage 1 first."
            )
            return False

        all_successful = True
        for dataset_dir in dataset_dirs:
            print(
                f"\n--- Building COLMAP databases for dataset: {dataset_dir.name} ---"
            )

            # NOTE: For undistorted images, we are using the 'left' views.
            configs = [
                {
                    "name": "Distorted Images",
                    "cam1_folder": dataset_dir / "distorted_images_cam1",
                    "cam2_folder": dataset_dir / "distorted_images_cam2",
                    "cam1_type": "distorted",
                    "cam2_type": "distorted",
                    "cam1_model": "OPENCV_FISHEYE",
                    "cam2_model": "OPENCV_FISHEYE",
                },
                {
                    "name": "Undistorted Images (Left Views)",
                    "cam1_folder": dataset_dir / "undistorted_cam1_left",
                    "cam2_folder": dataset_dir / "undistorted_cam2_left",
                    "cam1_type": "undistorted",
                    "cam2_type": "undistorted",
                    "cam1_model": "SIMPLE_RADIAL",
                    "cam2_model": "SIMPLE_RADIAL",
                },
            ]

            for config in configs:
                print(f"\nBuilding for {config['name']}...")

                retriever = ColmapImageRetriever(
                    cam1_folder=config["cam1_folder"],
                    cam2_folder=config["cam2_folder"],
                    cam1_camera_model=config["cam1_model"],
                    cam2_camera_model=config["cam2_model"],
                    cam1_type=config["cam1_type"],
                    cam2_type=config["cam2_type"],
                )

                retriever.initialize_databases(force_rebuild=force_rebuild)

        print("\nAll COLMAP databases built successfully")
        return True

    except Exception as e:
        print(f"Error building COLMAP databases: {e}")
        import traceback

        traceback.print_exc()
        return False


def analyze_retrieval_accuracy(sample_rate: int = 10):
    """
    Stage 7: Analyze image retrieval accuracy for each dataset.
    """
    print_stage_header(
        7, "RETRIEVAL ACCURACY ANALYSIS", "Analyzing location errors for each dataset"
    )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retrieval_accuracy_results_{timestamp}.txt"

    results_output = []
    results_output.append("=" * 80)
    results_output.append("RETRIEVAL ACCURACY ANALYSIS RESULTS")
    results_output.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results_output.append(f"Sample rate: Every {sample_rate}th image")

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "distorted_images_cam1").is_dir()
            ]
        )

        if not dataset_dirs:
            print("No dataset folders found. Please run Stage 1 first.")
            return False

        for dataset_dir in dataset_dirs:
            results_output.append("\n" + "=" * 80)
            results_output.append(f"DATASET: {dataset_dir.name}")
            results_output.append("=" * 80)

            orientation_csv_path = dataset_dir / "gps_interpolated_with_orientation.csv"
            if not orientation_csv_path.is_file():
                print(
                    f"❌ Orientation CSV not found for {dataset_dir.name}. Skipping analysis."
                )
                results_output.append(
                    "Orientation CSV not found. Skipping analysis for this dataset."
                )
                continue

            print(
                f"Loading GPS data for {dataset_dir.name} from {orientation_csv_path}..."
            )
            gps_df = pd.read_csv(orientation_csv_path)
            gps_lookup = {
                row["filename"]: (row["easting"], row["northing"])
                for _, row in gps_df.iterrows()
            }
            results_output.append(f"Loaded GPS data for {len(gps_lookup)} images.")

            analysis_configs = [
                {
                    "name": "Distorted Images Analysis",
                    "cam1_folder": dataset_dir / "distorted_images_cam1",
                    "cam2_folder": dataset_dir / "distorted_images_cam2",
                    "cam1_type": "distorted",
                    "cam2_type": "distorted",
                    "cam1_model": "OPENCV_FISHEYE",
                    "cam2_model": "OPENCV_FISHEYE",
                    "query_type": "distorted",
                },
                {
                    "name": "Undistorted Images Analysis (Left Views)",
                    "cam1_folder": dataset_dir / "undistorted_cam1_left",
                    "cam2_folder": dataset_dir / "undistorted_cam2_left",
                    "cam1_type": "undistorted",
                    "cam2_type": "undistorted",
                    "cam1_model": "SIMPLE_RADIAL",
                    "cam2_model": "SIMPLE_RADIAL",
                    "query_type": "undistorted",
                },
            ]

            for config in analysis_configs:
                print(f"\n--- {config['name']} for {dataset_dir.name} ---")
                results_output.append(f"\n--- {config['name']} ---")

                for cam_name, folder in [
                    ("cam1", config["cam1_folder"]),
                    ("cam2", config["cam2_folder"]),
                ]:
                    if not folder.is_dir():
                        print(f"Folder not found, skipping: {folder}")
                        continue

                    print(f"\nAnalyzing {cam_name}...")
                    results_output.append(f"\nAnalyzing {cam_name}...")

                    all_images = sorted(
                        list(folder.glob("*.jpg")) + list(folder.glob("*.png")),
                        key=lambda x: int(
                            get_original_numbered_file_name(x.name).split(".")[0]
                        ),
                    )

                    if not all_images:
                        print(f"No images found in {folder}")
                        results_output.append(f"No images found in {folder}")
                        continue

                    sample_images = all_images[::sample_rate]
                    print(
                        f"Found {len(all_images)} images, sampling {len(sample_images)}"
                    )

                    all_errors, missing_gps, no_matches, processed = [], 0, 0, 0

                    for i, query_image in enumerate(sample_images):
                        original_filename = get_original_numbered_file_name(
                            query_image.name
                        )
                        if original_filename not in gps_lookup:
                            missing_gps += 1
                            continue

                        query_easting, query_northing = gps_lookup[original_filename]

                        matches = retrieve_top_matches(
                            query_image_path=query_image,
                            cam1_folder=config["cam1_folder"],
                            cam2_folder=config["cam2_folder"],
                            cam1_camera_model=config["cam1_model"],
                            cam2_camera_model=config["cam2_model"],
                            cam1_type=config["cam1_type"],
                            cam2_type=config["cam2_type"],
                            query_image_type=config["query_type"],
                            top_n=5,
                        )

                        if not matches:
                            no_matches += 1
                            continue

                        processed += 1
                        for match_name, _ in matches:
                            original_match_filename = get_original_numbered_file_name(
                                match_name
                            )
                            if original_match_filename in gps_lookup:
                                match_easting, match_northing = gps_lookup[
                                    original_match_filename
                                ]
                                error = np.sqrt(
                                    (query_easting - match_easting) ** 2
                                    + (query_northing - match_northing) ** 2
                                )
                                all_errors.append(error)

                    print(f"  Images with matches: {processed}/{len(sample_images)}")
                    results_output.append(
                        f"  Images with matches: {processed}/{len(sample_images)}"
                    )
                    if all_errors:
                        mean_error = np.mean(all_errors)
                        print(f"  Mean Error: {mean_error:.2f} meters")
                        results_output.append(f"  Mean Error: {mean_error:.2f} meters")

        with open(output_file, "w") as f:
            f.write("\n".join(results_output))

        print(f"\nResults saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error during retrieval analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_lidar_skyline_generation():
    """
    Stage 8: Generate a LiDAR skyline database.
    This stage first creates a grid of observer points and then generates a
    skyline mask and data for each point using the raw LAZ files.
    """
    print_stage_header(
        8,
        "LIDAR SKYLINE DATABASE GENERATION",
        "Creating a grid and then generating skyline data from LAZ files.",
    )

    try:
        # This grid file will be used as input for the skyline generation.
        grid_output_path = constants.OBSERVER_GRID_PATH
        print(f"\n Generating Observer Grid")
        # print(f"Grid file will be saved to: {grid_output_path}")

        create_and_save_grid(
            start_coords=(
                611597.725,
                6458055.924,
            ),  # Bottom-left corner (Easting, Northing)
            spacing_meters=50.0,  # Distance between points
            num_points_x=3,  # Number of columns
            num_points_y=3,  # Number of rows
            output_path=grid_output_path,
        )

        print(f"\n Generating Skyline Database from Grid")
        output_db_directory = constants.DEM_OUTPUT_DIR / "skyline_database"
        viewpoint_z_height = 1.5  # Standard camera height in meters

        # Load the coordinates from the file we just created
        try:
            grid_df = pd.read_csv(grid_output_path)
            coords_to_process = list(zip(grid_df["easting"], grid_df["northing"]))
            print(
                f"Successfully loaded {len(coords_to_process)} coordinates from grid file."
            )
        except (FileNotFoundError, KeyError) as e:
            print(f" ERROR: Could not read the generated grid file: {e}")
            print("Aborting skyline generation.")
            return False

        # Run the database creation function
        create_skyline_database(
            coords_list=coords_to_process,
            base_z=viewpoint_z_height,
            output_directory=output_db_directory,
        )

        print("\n LiDAR skyline database generation completed successfully!")
        return True

    except Exception as e:
        print(f" An error occurred during LiDAR processing: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_skyline_fourier_encoder():
    """
    Stage 9: Select a dataset and create fingerprint databases in the project root
    under a new per-dataset directory structure.
    """
    print_stage_header(
        9,
        "SKYLINE FINGERPRINTING",
        "Generating 'global' and 'windowed' fingerprint databases.",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "binary_panorama").is_dir()
            ]
        )

        if not dataset_dirs:
            print("No binary panorama folders found. Please run Stage 5 first.")
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to fingerprint:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")
            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print(
                        f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(f"\n--- Fingerprinting skylines for dataset: {selected_dir.name} ---")

        input_binary_dir = selected_dir / "binary_panorama"
        dataset_name = selected_dir.name

        project_root = Path(__file__).parent
        dataset_db_dir = project_root / "databases" / dataset_name

        all_successful = True

        print("\nRunning 'global' fingerprinting strategy...")
        output_global_dir = dataset_db_dir / "global_fourier_fingerprints"
        global_output_path = output_global_dir / "fingerprints.npz"

        exit_code_global = fourier_skyline_encoder.create_fingerprints_for_directory(
            input_dir=input_binary_dir,
            output_filepath=global_output_path,
            strategy="global",
        )
        if exit_code_global != 0:
            all_successful = False
            print("Global fingerprinting failed.")

        print("\nRunning 'windowed' fingerprinting strategy...")
        output_windowed_dir = dataset_db_dir / "windowed_fourier_fingerprints"
        windowed_output_path = output_windowed_dir / "fingerprints.npz"

        exit_code_windowed = fourier_skyline_encoder.create_fingerprints_for_directory(
            input_dir=input_binary_dir,
            output_filepath=windowed_output_path,
            strategy="windowed",
        )
        if exit_code_windowed != 0:
            all_successful = False
            print("Windowed fingerprinting failed.")

        return all_successful

    except Exception as e:
        print(f"An error occurred during skyline fingerprinting: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_matching_analysis():
    """
    Stage 10: Perform a self-matching analysis on a fingerprint database,
    showing detailed per-window scores for windowed strategies.
    """
    print_stage_header(
        10,
        "FINGERPRINT SELF-MATCHING ANALYSIS",
        "Matching real panoramas against each other to test fingerprint quality.",
    )

    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"

        available_fp_files = sorted(list(db_root_dir.glob("*/*/*.npz")))
        if not available_fp_files:
            print(
                f"No fingerprint files found in '{db_root_dir}'. Please run Stage 9 first."
            )
            return False

        selected_fp_path = None
        while True:
            print("\nPlease select a fingerprint file to analyze:")
            for i, file_path in enumerate(available_fp_files):
                print(f"  [{i}]: {file_path.relative_to(project_root)}")
            choice_str = input("Enter the number of the file: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_fp_files):
                    selected_fp_path = available_fp_files[choice_idx]
                    break
                else:
                    print("❌ Invalid choice.")
            except ValueError:
                print("❌ Please enter a number.")

        if "windowed" in str(selected_fp_path):
            num_windows = 4
            print(f"\nWindowed strategy detected (using {num_windows} windows).")
        else:
            num_windows = 1
            print("\nGlobal strategy detected (using 1 window).")

        results = fourier_skyline_matcher.run_self_matching(
            fingerprint_path=selected_fp_path,
            num_samples=10,
            top_n=5,
            num_windows=num_windows,
        )

        if results:
            print("\n--- Top 5 Self-Matching Results ---")
            for result in results:
                print(f"\n  Query: {result['query']}")
                if result["top_matches"]:
                    for match in result["top_matches"]:
                        scores_str = ", ".join(
                            [f"{s:.3f}" for s in match["window_scores"]]
                        )
                        print(
                            f"    -> Match: {match['match_key']:<25} | Avg Score: {match['avg_score']:.4f} | Window Scores: [{scores_str}]"
                        )
                else:
                    print("    -> No other images found to match against.")
        else:
            print("Analysis could not be completed.")

        return True

    except Exception as e:
        print(f"An error occurred during matching: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Fisheye image processing pipeline")
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        type=int,
        default=[],
        help="Stages to skip (e.g., --skip-stages 1 2)",
    )
    parser.add_argument(
        "--only-stages",
        nargs="+",
        type=int,
        default=[],
        help="Only run specific stages (e.g., --only-stages 3 4)",
    )
    parser.add_argument(
        "--rebuild-colmap",
        action="store_true",
        help="Force rebuild of COLMAP databases",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="Sample rate for analysis (default: every 10th image)",
    )

    args = parser.parse_args()
    # Define all pipeline stages
    stages = {
        1: ("Split Images", run_image_splitting),
        2: ("Undistort Fisheye", run_fisheye_undistortion),
        3: ("Process GPS/Orientation", run_gps_orientation_processing),
        4: ("Stitch Panoramas", run_panorama_stitching),
        5: ("Segment Skylines", run_skyline_segmentation),
        6: (
            "Build COLMAP Databases",
            lambda: build_colmap_databases(args.rebuild_colmap),
        ),
        7: (
            "Analyze Retrieval Accuracy",
            lambda: analyze_retrieval_accuracy(args.sample_rate),
        ),
        8: ("Generate LiDAR Skylines", run_lidar_skyline_generation),
        9: ("Fourier Encode Binary Skylines", run_skyline_fourier_encoder),
        10: (
            "Fourier Skyline Matching",
            run_matching_analysis,
        ),  # TODO: This is just the start, later we want to implement encoding and matching for CNN feature extractions
    }

    # Determine which stages to run
    if args.only_stages:
        stages_to_run = args.only_stages
    else:
        stages_to_run = [s for s in stages.keys() if s not in args.skip_stages]

    print_separator("FISHEYE IMAGE PROCESSING PIPELINE")
    print(f"Stages to run: {sorted(stages_to_run)}")

    success = True
    start_time = time.time()

    for stage_num in sorted(stages_to_run):
        if stage_num in stages:
            stage_name, stage_func = stages[stage_num]
            stage_start = time.time()

            if not stage_func():
                print(f"\nStage {stage_num} failed. Stopping pipeline.")
                success = False
                break

            stage_duration = time.time() - stage_start
            print(f"Stage {stage_num} completed in {stage_duration:.1f} seconds")
        else:
            print(f"\nUnknown stage number: {stage_num}")

    total_duration = time.time() - start_time

    if success:
        print_separator("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_duration:.1f} seconds")
    else:
        print_separator("PIPELINE FAILED")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
