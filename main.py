import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import constants
from AI import evaluate_model, roll_angle_estimator
from AI.roll_angle_estimator import RollEstimatorCNN
from DEM.DEM_skyline import (
    check_file_relevance_from_json,
    create_and_save_grid,
    find_laz_files,
    generate_skyline_for_single_point,
    generate_skyline_mask_array,
    init_worker,
    process_viewpoint_and_file,
    visualize_comparison,
    visualize_single_skyline,
)
from image_processing import split_stacked_images
from image_processing.create_panoramas import build_panoramas
from image_processing.fisheye_undistorter import FisheyeUndistorter
from image_retrieval.colmap_naive_image_retrieval import (
    ColmapImageRetriever,
    retrieve_top_matches,
)
from skyline_processing import (
    cnn_skyline_encoder,
    fourier_skyline_encoder,
    skyline_matcher,
)
from skyline_processing.segformer_segmenter import SemanticSegmenter
from utils import orientation_path_from_gps
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


def run_fisheye_segmentation():
    """
    Stage 2: Segment the raw, distorted fisheye images to create binary masks and overlays.
    """
    print_stage_header(
        2,
        "FISHEYE IMAGE SEGMENTATION",
        "Generating binary masks and overlays from original distorted fisheye images.",
    )

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
            print("ERROR: No distorted image folders found. Please run Stage 1 first.")
            return False

        segmenter = SemanticSegmenter()
        if not segmenter.model:
            print("ERROR: Could not initialize SemanticSegmenter model.")
            return False

        all_successful = True
        for dataset_dir in dataset_dirs:
            print(f"\nSegmenting fisheye images for dataset: {dataset_dir.name}")

            for i in [1, 2]:
                input_dir = dataset_dir / f"distorted_images_cam{i}"
                output_mask_dir = dataset_dir / f"segmented_fisheye_cam{i}"
                output_overlay_dir = dataset_dir / f"overlay_segmented_fisheye_cam{i}"

                if not input_dir.is_dir() or not any(input_dir.iterdir()):
                    print(f"Skipping cam{i}: No images found in {input_dir}")
                    continue

                print(f"Processing {input_dir.name}...")

                exit_code = segmenter.process_directory(
                    input_dir=input_dir,
                    output_dir_black_white=output_mask_dir,
                    output_dir_overlay=output_overlay_dir,  # Pass the new overlay path
                )

                if exit_code != 0:
                    print(f"ERROR: Fisheye segmentation failed for {input_dir.name}.")
                    all_successful = False

        if all_successful:
            print("\nFisheye segmentation completed successfully for all datasets.")
        else:
            print("\nFisheye segmentation failed for one or more directories.")

        return all_successful

    except Exception as e:
        print(f"An error occurred during fisheye segmentation: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_roll_angle_training():
    """
    Stage 3: Train a CNN to estimate roll angle from segmented fisheye images.
    """
    print_stage_header(
        3,
        "ROLL ANGLE ESTIMATOR TRAINING",
        "Training a CNN on annotated, segmented fisheye images.",
    )

    try:
        project_root = Path(__file__).parent

        # Define the paths for the training data and model output
        annotation_csv_path = project_root / "AI" / "roll_annotations.csv"
        model_save_path = project_root / "AI" / "models" / "roll_estimator_v1.pth"

        if not annotation_csv_path.is_file():
            print(f"ERROR: Annotation file not found at '{annotation_csv_path}'")
            return False

        # Call the main training function from our new file
        exit_code = roll_angle_estimator.train_roll_estimator(
            images_root_dir=constants.IMAGES_DIR,
            annotation_csv=annotation_csv_path,
            model_save_path=model_save_path,
            num_epochs=30,  # TODO: Experiment with this
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_roll_model_evaluation():
    """
    Stage 4: Evaluate the trained roll angle estimation model on the validation set.
    """
    print_stage_header(
        4,
        "ROLL ANGLE MODEL EVALUATION",
        "Calculating error metrics and generating plots for the trained model.",
    )

    try:
        project_root = Path(__file__).parent

        # Define the paths for the data, trained model, and output plot
        images_root_dir = project_root / "images"
        annotation_csv_path = project_root / "AI" / "roll_annotations.csv"
        model_path = project_root / "AI" / "models" / "roll_estimator_v1.pth"
        plot_save_path = (
            project_root / "AI" / "evaluation_results" / "true_vs_predicted.png"
        )

        if not model_path.is_file():
            print(f"ERROR: Trained model not found at '{model_path}'")
            print("       Please run the training stage first.")
            return False

        # The transform must be identical to the one used during training
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        exit_code = evaluate_model.run_evaluation(
            images_root_dir=images_root_dir,
            annotation_csv=annotation_csv_path,
            model_path=model_path,
            plot_save_path=plot_save_path,
            transform=transform,
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_interactive_roll_prediction():
    """
    Stage 5: Manually test the roll angle model on a single, hardcoded image path.
    """
    print_stage_header(
        5,
        "SINGLE IMAGE ROLL PREDICTION",
        "Provide a path inside the function to get a roll angle prediction.",
    )

    try:
        image_to_test_path = (
            "images/dataset2/segmented_fisheye_cam1/binary_distorted_cam1_0447.png"
        )

        project_root = Path(__file__).parent
        model_path = project_root / "AI" / "models" / "roll_estimator_v1.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if the model exists
        if not model_path.is_file():
            print(f"ERROR: Trained model not found at '{model_path}'")
            print("       Please run the training stage first.")
            return False

        # Check if the test image exists
        image_path = Path(image_to_test_path)
        if not image_path.is_file():
            print(f"ERROR: Test image not found at '{image_path}'")
            print(
                "       Please update the 'image_to_test_path' variable in the script."
            )
            return False

        # 1. Load the model structure and weights
        model = RollEstimatorCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Loaded trained model on {device}.")

        # 2. Define the exact same transformations used during training
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        print(f"Processing image: {image_path.name}")

        # 3. Load and process the single image
        with Image.open(image_path).convert("L") as img:
            image_tensor = transform(img).unsqueeze(0).to(device)

        # 4. Get the prediction
        with torch.no_grad():
            predicted_angle = model(image_tensor).item()

        # 5. Display the result
        print("\nPrediction Result")
        print(f"Predicted Roll Angle: {predicted_angle:.2f} degrees")

        return True

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback

        traceback.print_exc()
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

# TODO: Updated idea! Perhaps we can detect the difference for where the islands align after building panoramas and segmenting them, and then we could use the difference in pixel alignment to find an angle that we want to roll them down by. We need to save this angle in a file, then we can make another pass with fisheye undistorter, with new rolls.

# TODO: Another method that would work is to interpolate the path of the boat with cubic splines and hope that is accurate enough. After that we can for the locations where the pictures are taken anotate the angle offset from a baseline straight path angle. These anotated angles in a relation to the amount of curve in the path the boat takes can be used to again interpolate what we believe to be the roll angle to curve magnitude. Im not sure if this method or the one using pixel offset between islands is best. The method above could sometimes give errors if the islands are not meant to align, which could happen since we are losing some information between the cameras.


def run_fisheye_undistortion():
    """
    Stage 6: Undistort fisheye images for all datasets.
    - Finds each dataset folder in the images directory.
    - Applies roll and horizon bend correction and a symmetrical vertical pitch correction to both cameras.
    """
    print_stage_header(
        6,
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
                        f"\nInvalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\nInvalid input. Please enter a number.")

        print(f"\nProcessing dataset: {selected_dir.name}")

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
    Stage 7: Interactively select a dataset and process its GPS data to calculate orientation.
    """
    print_stage_header(
        7,
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
                        f"\nInvalid input. Please enter a number between 0 and {len(available_datasets) - 1}."
                    )
            except ValueError:
                print("\nInvalid input. Please enter a number.")

        print(f"\nProcessing GPS data for: {selected_source_dir.name}")

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
        print(f"\nERROR: Could not find file: {e}")
        return False
    except Exception as e:
        print(f"\nAn error occurred during GPS orientation processing: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_panorama_stitching():
    """
    Stage 8: Interactively select a dataset and stitch its four rectilinear views into panoramas.
    """
    print_stage_header(
        8,
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
                        f"\nInvalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\nInvalid input. Please enter a number.")

        print(f"\nStitching panoramas for dataset: {selected_dir.name}")

        # Dynamically find the orientation CSV for the selected dataset
        orientation_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        if not orientation_csv_path.is_file():
            print(
                f"Orientation CSV not found for {selected_dir.name} at '{orientation_csv_path}'."
            )
            print("   Please run Stage 3 for this dataset first.")
            return False

        # Define input and output paths for the selected dataset
        cam1_left_dir = selected_dir / "undistorted_cam1_left"
        cam1_right_dir = selected_dir / "undistorted_cam1_right"
        cam2_left_dir = selected_dir / "undistorted_cam2_left"
        cam2_right_dir = selected_dir / "undistorted_cam2_right"
        output_panorama_dir = selected_dir

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
    Stage 9: Interactively select a dataset and use segformer-b5 to create
    binary skyline masks from its panoramas.
    """
    print_stage_header(
        9,
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
                        f"\nInvalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\nInvalid input. Please enter a number.")

        print(f"\nSegmenting skylines for selected dataset: {selected_dir.name}")

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
            print(f"\nSkyline segmentation failed for {selected_dir.name}!")
            return False

    except Exception as e:
        print(f"An error occurred during skyline segmentation: {e}")
        import traceback

        traceback.print_exc()
        return False


def build_colmap_databases(force_rebuild: bool = False):
    """
    Stage 10: Build COLMAP databases for image retrieval for each dataset.
    """
    print_stage_header(
        10, "BUILD COLMAP DATABASES", "Building feature databases for each dataset"
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

        for dataset_dir in dataset_dirs:
            print(f"\nBuilding COLMAP databases for dataset: {dataset_dir.name}")

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
    Stage 11: Analyze image retrieval accuracy for each dataset.
    """
    print_stage_header(
        11, "RETRIEVAL ACCURACY ANALYSIS", "Analyzing location errors for each dataset"
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
                    f"Orientation CSV not found for {dataset_dir.name}. Skipping analysis."
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
                print(f"\n{config['name']} for {dataset_dir.name}")
                results_output.append(f"\n{config['name']}")

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
                            get_original_numbered_file_name(x.name).split(".")[0]  # type: ignore
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


def run_single_point_validation():
    """
    Stage 12: Generate a DEM skyline for a single, user-selected coordinate
    and compare it visually against the real segmented panorama.
    """
    print_stage_header(
        12,
        "SINGLE-POINT VALIDATION",
        "Visually comparing a single DEM skyline against a real panorama.",
    )
    try:
        # 1. Interactively select a dataset
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir()
                and (d / "gps_interpolated_with_orientation.csv").is_file()
            ]
        )
        if not dataset_dirs:
            print("No datasets with 'gps_interpolated_with_orientation.csv' found.")
            return False

        selected_dir = dataset_dirs[0]
        print(f"Using dataset: {selected_dir.name}")

        # 2. Select a specific data point from the CSV
        gps_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        gps_df = pd.read_csv(gps_csv_path)

        print("\nPlease select a data point to validate from the table below:")
        print(gps_df.head(10).to_string())  # Show first 10 options
        choice_str = input(f"Enter the index (0-{len(gps_df) - 1}): ")
        choice_idx = int(choice_str)

        selected_point = gps_df.iloc[choice_idx]
        easting, northing = selected_point["easting"], selected_point["northing"]
        original_filename = selected_point["filename"]  # e.g., '0801.jpg'
        viewpoint_coords = (easting, northing, 1.5)  # Assume 1.5m height

        # 3. Generate the DEM skyline for this single point
        dem_mask = generate_skyline_for_single_point(viewpoint_coords)

        # 4. Load the corresponding real segmented panorama
        image_number = Path(original_filename).stem  # '0801'
        real_mask_name = f"binary_panorama_{image_number}.png"
        real_mask_path = selected_dir / "binary_panorama" / real_mask_name

        real_mask = np.array(Image.open(real_mask_path))

        # 5. Display the side-by-side comparison
        point_id = f"{easting:.0f}_{northing:.0f}"
        visualize_comparison(dem_mask, real_mask, point_id, real_mask_name)

        return True

    except Exception as e:
        print(f"An error occurred during single-point validation: {e}")
        import traceback

        traceback.print_exc()
        return False


# ! Warning: This method will take an enormous amount of time to run. If we want to have a finer grid than 25 meters, a good method now would be to shift the start and end point by 12,5 meters, and then run another 25 meter grid calculation.


# ! Important though! if we do the above, we need to make sure that we dont overwrite the old database, but that we instead concatenate to it!
# In main.py, replace the whole function


def run_lidar_skyline_generation():
    """
    Stage 13: Generate a LiDAR skyline database using a resilient, batch-based architecture.
    """
    print_stage_header(13, "LIDAR SKYLINE MASK GENERATION (BATCHED & RESILIENT)")

    try:
        BATCH_SIZE = 100  # Process 100 viewpoints per batch
        project_root = Path(__file__).resolve().parent
        output_dir = project_root / "databases" / "dem_database"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "skyline_masks.npz"

        grid_df = create_and_save_grid(
            start_coords=(611597.725, 6458055.924),
            end_coords=(614547.264, 6463806.450),
            spacing_meters=25.0,
        )
        # Create a lookup from point_id to coordinates
        grid_df["point_id"] = grid_df.apply(
            lambda row: f"dem_vp_{row['easting']:.0f}_{row['northing']:.0f}", axis=1
        )
        coord_lookup = grid_df.set_index("point_id")[["easting", "northing"]].to_dict(
            "index"
        )
        all_point_ids = grid_df["point_id"].tolist()
        all_laz_files = find_laz_files()

        # 2. RESUME LOGIC
        processed_ids = set()
        if output_path.exists():
            with np.load(output_path) as existing_data:
                processed_ids = set(existing_data.keys())
            print(
                f"Found existing database with {len(processed_ids)} processed points."
            )

        point_ids_to_process = [
            pid for pid in all_point_ids if pid not in processed_ids
        ]
        if not point_ids_to_process:
            print(
                "\n✅ All points have already been processed. Database is up to date."
            )
            return True

        print(
            f"Total points to process: {len(point_ids_to_process)} / {len(all_point_ids)}"
        )

        # 3. BATCHING
        batches = [
            point_ids_to_process[i : i + BATCH_SIZE]
            for i in range(0, len(point_ids_to_process), BATCH_SIZE)
        ]
        num_processes = max(1, cpu_count() - 10)
        worker_config = {
            "SEARCH_RADIUS": constants.SEARCH_RADIUS,
            "ANGLE_STEP": constants.ANGLE_STEP,
            "MIN_DISTANCE": constants.MIN_DISTANCE,
        }

        # 4. MAIN BATCH PROCESSING LOOP
        for i, batch_point_ids in enumerate(batches):
            print_separator(f"Processing Batch {i + 1} / {len(batches)}")

            # A. Generate tasks just for this batch
            batch_tasks = []
            print("Preparing tasks for the current batch...")
            for point_id in tqdm(batch_point_ids, desc="Pre-calculating tasks"):
                coords = coord_lookup[point_id]
                viewpoint_coords = (coords["easting"], coords["northing"], 1.5)
                for laz_file in all_laz_files:
                    if check_file_relevance_from_json(laz_file, viewpoint_coords):
                        batch_tasks.append((point_id, viewpoint_coords, laz_file))

            if not batch_tasks:
                print("No relevant tasks for this batch. Skipping.")
                continue

            # B. Run workers for the batch
            print(
                f"Distributing {len(batch_tasks)} tasks across {num_processes} CPU cores..."
            )
            partial_results = []
            with Pool(
                processes=num_processes,
                initializer=init_worker,
                initargs=(worker_config,),
            ) as pool:
                iterator = pool.imap_unordered(process_viewpoint_and_file, batch_tasks)
                for result in tqdm(
                    iterator, total=len(batch_tasks), desc="Processing LAZ files"
                ):
                    if result and result[1] is not None:
                        partial_results.append(result)

            # C. Aggregate and generate masks for the batch
            batch_skyline_data = {}
            for point_id, (azimuths, elevations) in partial_results:
                if point_id not in batch_skyline_data:
                    num_bins = int(360 / constants.ANGLE_STEP)
                    batch_skyline_data[point_id] = np.full(num_bins, 0.0)

                df = pd.DataFrame({"azimuth": azimuths, "elevation": elevations})
                num_bins = len(batch_skyline_data[point_id])
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
                batch_skyline_data[point_id] = np.maximum(
                    batch_skyline_data[point_id], new_maxes
                )

            batch_final_masks = {}
            for point_id, elevations in batch_skyline_data.items():
                num_bins = len(elevations)
                angles = np.linspace(0, 360, num_bins, endpoint=False)
                batch_final_masks[point_id] = generate_skyline_mask_array(
                    angles,
                    elevations,
                    constants.PANORAMA_IMG_WIDTH,
                    constants.PANORAMA_IMG_HEIGHT,
                )

            # D. Incremental Save
            print(
                f"Saving results for {len(batch_final_masks)} points from this batch..."
            )
            # Load existing data, update it, and save back
            existing_data = {}
            if output_path.exists():
                existing_data = dict(np.load(output_path))

            existing_data.update(batch_final_masks)
            np.savez_compressed(output_path, **existing_data)
            print(f"Database now contains {len(existing_data)} total masks.")

        print(f"\n✅ LiDAR skyline mask generation complete. Saved to '{output_path}'.")
        return True

    except Exception as e:
        print(f"An error occurred during LiDAR processing: {e}")
        import traceback

        traceback.print_exc()
        return False


# TODO: Maybe one thing we want to do, since the panoramas we have lose some information in the borders where the two camera images connect, perhaps before rotating the panoramas in north normalized direction, just like the dems are, we keep track of how much they need to be rotated, and then where the edges between the front and back facing images are, and cut out a couple of percent HFOV from the DEM simulated image there as well. This is because the DEM image has perfect 360deg coverage while our panoramas has maybe 2x175deg or something similar in practice, even though it should be 2x180deg.
def run_dem_skyline_encoding():
    """
    Stage 14: Load a DEM skyline mask database and create all fingerprint types.
    """
    print_stage_header(
        14,
        "DEM SKYLINE ENCODING",
        "Generates all fingerprint types from a pre-computed DEM mask database.",
    )

    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"
        mask_db_path = db_root_dir / "dem_database" / "skyline_masks.npz"

        if not mask_db_path.is_file():
            print(f" DEM skyline mask database not found at '{mask_db_path}'.")
            print("   Please run Stage 8 first.")
            return False

        print(f"Loading DEM skyline masks from {mask_db_path}...")
        skyline_mask_database = np.load(mask_db_path)
        print(f"Loaded {len(skyline_mask_database.files)} simulated skylines.")

        # Define the output base directory
        dataset_db_dir = db_root_dir / "dem_database"

        # Fourier encoding
        for strategy in ["global", "windowed"]:
            fourier_fps = fourier_skyline_encoder.create_fingerprints_from_database(
                skyline_mask_database, strategy=strategy
            )
            if fourier_fps:
                output_path = (
                    dataset_db_dir
                    / f"fourier_{strategy}_fingerprints"
                    / "fingerprints.npz"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, **fourier_fps)
                print(f"Saved DEM Fourier ({strategy}) fingerprints to {output_path}")

        # CNN encoding
        cnn_fps = cnn_skyline_encoder.create_cnn_fingerprints_from_database(
            skyline_mask_database
        )
        if cnn_fps:
            output_path = dataset_db_dir / "cnn_fingerprints" / "fingerprints.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **cnn_fps)
            print(f"Saved DEM CNN fingerprints to {output_path}")

        print("\nDEM skyline encoding complete.")
        return True

    except Exception as e:
        print(f"An error occurred during DEM encoding: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_skyline_fourier_encoder():
    """
    Stage 15: Select a dataset and create fingerprint databases in the project root
    under a new per-dataset directory structure.
    """
    print_stage_header(
        15,
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
                        f"\n Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\n Invalid input. Please enter a number.")

        print(f"\nFingerprinting skylines for dataset: {selected_dir.name}")

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
    Stage 16: Perform a self-matching analysis on a fingerprint database,
    showing detailed per-window scores for windowed strategies.
    """
    print_stage_header(
        16,
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
                    print(" Invalid choice.")
            except ValueError:
                print(" Please enter a number.")

        if "windowed" in str(selected_fp_path):
            num_windows = 4
            print(f"\nWindowed strategy detected (using {num_windows} windows).")
        else:
            num_windows = 1
            print("\nGlobal strategy detected (using 1 window).")

        results = skyline_matcher.run_self_matching(
            fingerprint_path=selected_fp_path,
            num_samples=10,
            top_n=5,
            num_windows=num_windows,
        )

        if results:
            print("\nTop 5 Self-Matching Results")
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


def run_cnn_encoding():
    """
    Stage 17: Select a dataset and create CNN-based fingerprints for its binary panoramas.
    """
    print_stage_header(
        17,
        "CNN SKYLINE ENCODING",
        "Generating rich feature vectors using a pre-trained CNN.",
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
            print("\nPlease select a dataset to encode with the CNN:")
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
                        f"\nInvalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}."
                    )
            except ValueError:
                print("\nInvalid input. Please enter a number.")

        print(f"\nEncoding skylines for dataset: {selected_dir.name}")

        input_binary_dir = selected_dir / "binary_panorama"
        dataset_name = selected_dir.name

        # Define the new output directory structure
        project_root = Path(__file__).parent
        output_dir = project_root / "databases" / dataset_name / "cnn_fingerprints"
        output_path = output_dir / "fingerprints.npz"

        exit_code = cnn_skyline_encoder.create_cnn_fingerprints_for_directory(
            input_dir=input_binary_dir, output_filepath=output_path
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during CNN encoding: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_cnn_matching_analysis():
    """
    Stage 18: Perform a self-matching analysis on a CNN-based fingerprint
    database to validate the encoding method.
    """
    print_stage_header(
        18,
        "CNN FINGERPRINT SELF-MATCHING ANALYSIS",
        "Matching real panoramas against each other using CNN features.",
    )

    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"

        available_fp_files = sorted(list(db_root_dir.glob("*/cnn_fingerprints/*.npz")))
        if not available_fp_files:
            print(
                f"No CNN fingerprint files found in '{db_root_dir}/*/cnn_fingerprints/'."
            )
            print("   Please run Stage 11 (CNN Skyline Encoding) first.")
            return False

        selected_fp_path = None
        while True:
            print("\nPlease select a CNN fingerprint file to analyze:")
            for i, file_path in enumerate(available_fp_files):
                print(f"  [{i}]: {file_path.relative_to(project_root)}")
            choice_str = input("Enter the number of the file: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_fp_files):
                    selected_fp_path = available_fp_files[choice_idx]
                    break
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

        # The CNN encoder also uses 4 windows, so we can reuse the matcher perfectly
        results = skyline_matcher.run_self_matching(
            fingerprint_path=selected_fp_path, num_samples=10, top_n=5, num_windows=4
        )

        if results:
            print("\nTop 5 Self-Matching Results (CNN)")
            for result in results:
                print(f"\n  Query: {result['query']}")
                top_matches_list = result.get("top_matches", [])
                if top_matches_list:
                    for match in top_matches_list:
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
        print(f"An error occurred during CNN matching analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_dem_visualization():
    """
    Stage 19: Visualize a few sample DEM skylines using coordinates from a dataset's GPS file.
    """
    print_stage_header(
        19,
        "VISUALIZE DEM SKYLINE MASKS",
        "Displaying sample masks generated from LiDAR data.",
    )
    try:
        # 1. Check for the DEM database
        project_root = Path(__file__).resolve().parent
        dem_db_path = project_root / "databases" / "dem_database" / "skyline_masks.npz"
        if not dem_db_path.exists():
            print(f"DEM database not found at '{dem_db_path}'.")
            print("   Please run Stage 12 (LIDAR SKYLINE MASK GENERATION) first.")
            return False

        # 2. Interactively select a dataset to get coordinates from
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir()
                and (d / "gps_interpolated_with_orientation.csv").is_file()
            ]
        )
        if not dataset_dirs:
            print("No datasets with 'gps_interpolated_with_orientation.csv' found.")
            return False

        selected_dir = None
        while True:
            print(
                "\nPlease select a dataset to pull GPS coordinates from for visualization:"
            )
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")
            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

        # 3. Load CSV and find closest points in the DEM database
        gps_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        print(f"\nLoading coordinates from {gps_csv_path.name}...")
        gps_df = pd.read_csv(gps_csv_path)

        # Take up to 3 random sample points from the CSV
        num_samples = min(3, len(gps_df))
        sample_points = gps_df.sample(n=num_samples)
        print(f"\nAttempting to visualize {num_samples} points from the trajectory.")
        print(
            "Note: Finding the NEAREST grid point in the DEM database for each sample."
        )

        with np.load(dem_db_path) as data:
            db_keys = data.files
            # Parse 'dem_vp_easting_northing' keys into a numpy array of coordinates
            db_coords = np.array(
                [[int(k.split("_")[2]), int(k.split("_")[3])] for k in db_keys]
            )

            for _, row in sample_points.iterrows():
                query_coord = np.array([row["easting"], row["northing"]])
                # Find the index of the closest coordinate in the database
                distances = np.sum((db_coords - query_coord) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                closest_point_id = db_keys[closest_idx]

                # Call the visualization function
                visualize_single_skyline(dem_db_path, closest_point_id)

        print("\nDEM Visualization complete.")
        return True

    except Exception as e:
        print(f"An error occurred during DEM visualization: {e}")
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
        2: ("Segment Fisheye For Roll Network", run_fisheye_segmentation),
        3: ("Train Roll Angle Estimator", run_roll_angle_training),
        4: ("Evaluate Roll Angle Model", run_roll_model_evaluation),
        5: ("Interactive Roll Prediction", run_interactive_roll_prediction),
        6: ("Undistort Fisheye", run_fisheye_undistortion),
        7: ("Process GPS/Orientation", run_gps_orientation_processing),
        8: ("Stitch Panoramas", run_panorama_stitching),
        9: ("Segment Skylines", run_skyline_segmentation),
        10: (
            "Build COLMAP Databases",
            lambda: build_colmap_databases(args.rebuild_colmap),
        ),
        11: (
            "Analyze Retrieval Accuracy",
            lambda: analyze_retrieval_accuracy(args.sample_rate),
        ),
        12: ("Single Point Validation", run_single_point_validation),
        13: ("Generate DEM Skyline Masks", run_lidar_skyline_generation),
        14: ("Encode DEM Skylines", run_dem_skyline_encoding),
        15: ("Encode REAL Skylines (Fourier)", run_skyline_fourier_encoder),
        16: ("Encode REAL Skylines (CNN)", run_cnn_encoding),
        17: ("Fourier Matching Analysis", run_matching_analysis),
        18: ("CNN Matching Analysis", run_cnn_matching_analysis),
        19: ("Visualize DEM Skylines", run_dem_visualization),
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
