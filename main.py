import argparse
import gc
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import constants
from AI import evaluate_model, roll_angle_estimator
from AI.plotting import create_cross_match_plot, create_results_plot
from AI.roll_angle_estimator import (
    RollEstimatorCNN,
    run_kfold_cross_validation,
    visualize_model_features,
)
from DEM.DEM_skyline import (
    calculate_skyline_similarity,
    check_file_relevance_from_json,
    create_and_save_grid,
    find_laz_files,
    generate_skyline_for_single_point,
    generate_skyline_mask_array,
    init_worker,
    process_tile_for_multiple_viewpoints,
    process_viewpoint_and_file,
    visualize_comparison,
    visualize_single_skyline,
)
from image_processing import split_stacked_images
from image_processing.create_panoramas import (
    build_panoramas,
    rotate_panorama_by_orientation,
)
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
    Stage 3: Train a single, unified CNN model on a pre-defined list of datasets.
    """
    print_stage_header(
        3,
        "UNIFIED ROLL ANGLE ESTIMATOR TRAINING",
        "Training one model on a combined dataset for better generalization.",
    )
    try:
        project_root = Path(__file__).parent
        annotation_csv_path = project_root / "AI" / "roll_annotations.csv"
        images_root = Path(constants.IMAGES_DIR)

        if not annotation_csv_path.is_file():
            print(f"ERROR: Annotation file not found at '{annotation_csv_path}'")
            return False

        # Add the paths to all the image folders you want to train on.
        training_folders = [
            # images_root / "dataset1" / "distorted_images_cam1",
            images_root / "dataset2" / "distorted_images_cam1",
            images_root / "dataset3" / "distorted_images_cam1",
        ]

        # Filter out any folders that don't actually exist
        valid_training_folders = [p for p in training_folders if p.is_dir()]
        if not valid_training_folders:
            print("None of the specified training folders exist. Aborting.")
            return False

        model_save_path = project_root / "AI" / "models" / "roll_estimator_combined.pth"

        exit_code = roll_angle_estimator.train_roll_estimator(
            image_dirs=valid_training_folders,  # Pass the list of folders
            annotation_csv=annotation_csv_path,
            model_save_path=model_save_path,
            num_epochs=120,
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_roll_angle_cross_validation():
    """
    Stage 3b: Run K-Fold Cross-Validation on the Roll Angle Estimator.
    """
    print_stage_header(
        "3b",
        "ROLL ANGLE ESTIMATOR K-FOLD CROSS-VALIDATION",
        "Rigorously evaluating model performance across multiple data splits.",
    )
    try:
        project_root = Path(__file__).parent
        annotation_csv_path = project_root / "AI" / "roll_annotations.csv"
        images_root = Path(constants.IMAGES_DIR)

        if not annotation_csv_path.is_file():
            print(f"ERROR: Annotation file not found at '{annotation_csv_path}'")
            return False

        # CONFIGURE YOUR TRAINING DATA HERE
        training_folders = [
            # images_root / "dataset1" / "distorted_images_cam1",
            images_root / "dataset2" / "distorted_images_cam1",
            images_root / "dataset3" / "distorted_images_cam1",
        ]
        valid_training_folders = [p for p in training_folders if p.is_dir()]
        if not valid_training_folders:
            print("None of the specified training folders exist. Aborting.")
            return False

        # Call the helper function that contains all the complex logic
        exit_code = run_kfold_cross_validation(
            image_dirs=valid_training_folders,
            annotation_csv=annotation_csv_path,
            k_folds=5,
            num_epochs=15,
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during cross-validation: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_roll_model_evaluation():
    """
    Stage 4: Evaluate the combined roll angle model on the validation set
    from the full training data.
    """
    print_stage_header(4, "ROLL ANGLE MODEL EVALUATION")
    try:
        project_root = Path(__file__).parent
        annotation_csv_path = project_root / "AI" / "roll_annotations.csv"
        images_root = Path(constants.IMAGES_DIR)

        model_path = project_root / "AI" / "models" / "roll_estimator_combined.pth"
        if not model_path.is_file():
            print(
                f"ERROR: Combined model not found at '{model_path}'. Please run Stage 3 first."
            )
            return False

        # Use the SAME list of folders that the model was trained on
        training_folders = [
            # images_root / "dataset1" / "distorted_images_cam1",
            images_root / "dataset2" / "distorted_images_cam1",
            images_root / "dataset3" / "distorted_images_cam1",
        ]

        valid_folders = [p for p in training_folders if p.is_dir()]
        if not valid_folders:
            print("None of the specified evaluation folders exist.")
            return False

        # This must be identical to the transform used during training
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        # END OF BLOCK

        plot_save_path = (
            project_root / "AI" / "evaluation_results" / "evaluation_combined_model.png"
        )

        exit_code = evaluate_model.run_evaluation(
            image_dirs=valid_folders,
            annotation_csv=annotation_csv_path,
            model_path=model_path,
            plot_save_path=plot_save_path,
            transform=transform,  # <<< PASS THE TRANSFORM HERE
        )
        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_interactive_roll_prediction():
    """
    Stage 5: Manually test the combined roll angle model on a single image.
    """
    print_stage_header(5, "SINGLE IMAGE ROLL PREDICTION")
    try:
        project_root = Path(__file__).parent

        # Path to the single combined model
        model_path = project_root / "AI" / "models" / "roll_estimator_combined.pth"
        if not model_path.is_file():
            print(
                f"ERROR: Combined model not found at '{model_path}'. Please run Stage 3 first."
            )
            return False

        # CONFIGURE THE TEST IMAGE HERE
        # Define the relative path to the image you want to test.
        image_path = (
            project_root
            / "images/dataset2/distorted_images_cam1/distorted_cam1_0447.jpg"
        )
        # END OF CONFIGURATION

        if not image_path.is_file():
            print(f"ERROR: Test image not found at '{image_path}'")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RollEstimatorCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded trained model '{model_path.name}' on {device}.")

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        print(f"Processing image: {image_path.name}")
        with Image.open(image_path).convert("L") as img:
            image_tensor = transform(img)

        # Apply the focus mask
        _, height, width = image_tensor.shape
        center_x, center_y = width // 2, height // 2
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        scale = width / 2.5
        dist_from_center = torch.sqrt(
            ((X - center_x) ** 2 + (Y - center_y) ** 2).float()
        )
        focus_mask = torch.exp(-(dist_from_center**2) / (2 * scale**2))
        image_tensor = image_tensor * focus_mask.unsqueeze(0)

        # Add batch dimension and send to device
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_angle = model(image_tensor).item()

        print("\nPrediction Result")
        print(f"Predicted Roll Angle: {predicted_angle:.2f} degrees")
        return True

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_feature_visualization():
    """
    Stage 6: Visualize the features learned by the combined Roll Angle Estimator CNN.
    """
    print_stage_header(6, "CNN FEATURE VISUALIZATION")
    try:
        project_root = Path(__file__).parent

        # Path to the single combined model
        model_path = project_root / "AI" / "models" / "roll_estimator_combined.pth"
        if not model_path.is_file():
            print(
                f"ERROR: Combined model not found at '{model_path}'. Please run Stage 3 first."
            )
            return False

        # CONFIGURE THE VISUALIZATION IMAGE HERE
        # Define the relative path to the image you want to visualize.
        sample_image_path = (
            project_root
            / "images/dataset2/distorted_images_cam1/distorted_cam1_0447.jpg"
        )
        # END OF CONFIGURATION

        if not sample_image_path.is_file():
            print(f"ERROR: Sample image not found at '{sample_image_path}'.")
            return False

        output_dir = project_root / "AI" / "evaluation_results"

        visualize_model_features(
            model_path=model_path,
            image_path=sample_image_path,
            output_dir=output_dir,
        )
        return True

    except Exception as e:
        print(f"An error occurred during feature visualization: {e}")
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
    Stage 7: Undistort fisheye images for all datasets.
    - Finds each dataset folder in the images directory.
    - Applies roll and horizon bend correction and a symmetrical vertical pitch correction to both cameras.
    """
    print_stage_header(
        7,
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
            f"\nFisheye undistortion for '{selected_dir.name}' finished successfully!"
        )
        return True

    except Exception as e:
        print(f"An error occurred during fisheye undistortion: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_gps_orientation_processing():
    """
    Stage 8: Interactively select a dataset and process its GPS data to calculate orientation.
    """
    print_stage_header(
        8,
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
            f"\nGPS orientation processing for '{selected_source_dir.name}' complete."
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
    Stage 9: Interactively select a dataset and stitch its four rectilinear views into panoramas.
    """
    print_stage_header(
        9,
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
            manual_rotation_offset_deg = 32  # TODO: Verify with better calculations
        else:
            manual_rotation_offset_deg = (
                36  # TODO: Change for the camera setup in dataset 2 and 3
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
    Stage 10: Interactively select a dataset and use segformer-b5 to create
    binary skyline masks from its panoramas.
    """
    print_stage_header(
        10,
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
                f"\nSkyline segmentation completed successfully for {selected_dir.name}!"
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
    Stage 11: Build COLMAP databases for image retrieval for each dataset.
    """
    print_stage_header(
        11, "BUILD COLMAP DATABASES", "Building feature databases for each dataset"
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
    Stage 12: Analyze image retrieval accuracy for each dataset.
    """
    print_stage_header(
        12, "RETRIEVAL ACCURACY ANALYSIS", "Analyzing location errors for each dataset"
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


def modify_dem_mask_for_fov_loss(
    dem_mask: np.ndarray,
    yaw_offset: float,
    orientation: float,
    manual_offset_deg: float,
    gap_width_degrees: float,
) -> np.ndarray:
    """
    Modifies a DEM skyline mask to simulate information loss from stitched panoramas.

    This function removes two vertical bands (gaps) from the DEM mask to mimic the
    missing field-of-view at the stitch seams of the real panoramas. The remaining
    parts of the mask are then stretched to fill the original image width.

    Args:
        dem_mask (np.ndarray): The original 360-degree DEM skyline mask.
        yaw_offset (float): The calculated yaw offset (in degrees) to align the
                            DEM mask with the real panorama.
        orientation (float): The camera orientation (in degrees from North) from GPS data.
        manual_offset_deg (float): A manual constant offset in degrees.
        gap_width_degrees (float): The total angular width of each gap to be removed.

    Returns:
        np.ndarray: The modified DEM mask with gaps removed and content stretched.
    """
    height, width = dem_mask.shape[:2]

    # --- 1. Calculate the center of the first gap ---
    # The total rotation is the sum of the boat's orientation, the fine-tuned yaw
    # offset, and any manual hardware offset.
    # We use modulo 360 to handle wrap-around angles.
    total_rotation_deg = (manual_offset_deg + yaw_offset + orientation) % 360.0

    # Convert the angle to a pixel column for the center of the first gap
    gap1_center_px = (total_rotation_deg / 360.0) * width

    # --- 2. Calculate the center of the second gap (180 degrees away) ---
    pixels_for_180_deg = width / 2
    gap2_center_px = (gap1_center_px + pixels_for_180_deg) % width

    # --- 3. Determine the columns to remove for both gaps ---
    gap_width_px = int(round((gap_width_degrees / 360.0) * width))
    half_gap_px = gap_width_px // 2

    # Create a mask of columns to KEEP, initially all True
    cols_to_keep = np.ones(width, dtype=bool)
    all_col_indices = np.arange(width)

    # Identify and mark columns to remove for the first gap
    start1 = int(round(gap1_center_px - half_gap_px))
    end1 = int(round(gap1_center_px + half_gap_px))
    indices_to_remove1 = (all_col_indices >= start1) & (all_col_indices < end1)
    # Handle wrap-around (if the gap crosses the 0-degree seam)
    if start1 < 0:
        indices_to_remove1 = (all_col_indices >= (start1 + width)) | (
            all_col_indices < end1
        )
    elif end1 > width:
        indices_to_remove1 = (all_col_indices >= start1) | (
            all_col_indices < (end1 - width)
        )

    cols_to_keep[indices_to_remove1] = False

    # Identify and mark columns to remove for the second gap
    start2 = int(round(gap2_center_px - half_gap_px))
    end2 = int(round(gap2_center_px + half_gap_px))
    indices_to_remove2 = (all_col_indices >= start2) & (all_col_indices < end2)
    # Handle wrap-around
    if start2 < 0:
        indices_to_remove2 = (all_col_indices >= (start2 + width)) | (
            all_col_indices < end2
        )
    elif end2 > width:
        indices_to_remove2 = (all_col_indices >= start2) | (
            all_col_indices < (end2 - width)
        )

    cols_to_keep[indices_to_remove2] = False

    # --- 4. Create the gapped mask and stretch it ---
    # Select only the columns we want to keep
    gapped_mask = dem_mask[:, cols_to_keep]

    # Resize the gapped mask back to the original width, effectively stretching it.
    # INTER_NEAREST is crucial to maintain the binary nature of the mask.
    modified_dem_mask = cv2.resize(
        gapped_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )

    return modified_dem_mask


def run_batch_validation():
    """
    Stage 13: For a selected dataset, generate a DEM skyline for every
    coordinate in its GPS file and save a comparison plot against the real panorama.

    **This version modifies the DEM skyline to account for FOV loss in the real
    panoramas before calculating similarity.**
    """
    print_stage_header(
        13,
        "BATCH VALIDATION (DEM vs. REAL)",
        "Generating a full set of comparison images for a dataset.",
    )
    try:
        # --- Constants for DEM Modification ---
        # Per your formula and description
        MANUAL_OFFSET_DEG = 16.0
        # Total degrees of information loss. Assuming 5 degrees per seam.
        # This means each gap will be 5 degrees wide.
        GAP_WIDTH_DEG = 5.0

        # --- 1. Select the dataset to process ---
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
                    print("Invalid input.")
            except ValueError:
                print("Invalid input.")

        #  2. Prepare for batch processing
        project_root = Path(__file__).resolve().parent
        gps_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        gps_df = pd.read_csv(gps_csv_path)
        all_laz_files = find_laz_files()

        output_dir = (
            project_root
            / "output_plots"
            / "DEM_vs_panorama_same_coordinates"
            / selected_dir.name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Comparison plots will be saved to: {output_dir}")

        offset_results = []

        #  3. Loop through every coordinate in the CSV file
        for selected_point in tqdm(
            gps_df.itertuples(),
            total=len(gps_df),
            desc=f"Processing {selected_dir.name}",
        ):
            easting, northing = selected_point.easting, selected_point.northing
            orientation_from_csv = selected_point.Orientation
            viewpoint_coords = (easting, northing, 1.5)
            point_id = f"validation_vp_{easting:.0f}_{northing:.0f}"

            original_filename = selected_point.filename
            image_number = Path(original_filename).stem
            real_mask_name = f"binary_panorama_{image_number}.png"
            real_mask_path = selected_dir / "binary_panorama_aligned" / real_mask_name

            if not real_mask_path.exists():
                tqdm.write(
                    f"Warning: Real panorama not found for {original_filename}. Skipping."
                )
                continue

            real_mask = np.array(Image.open(real_mask_path))

            relevant_files = [
                f
                for f in all_laz_files
                if check_file_relevance_from_json(f, viewpoint_coords)
            ]
            if not relevant_files:
                tqdm.write(
                    f"Warning: No relevant LAZ files for {original_filename}. Skipping."
                )
                continue

            tasks = [
                (laz_file, [(point_id, viewpoint_coords)])
                for laz_file in relevant_files
            ]
            num_processes = max(1, cpu_count() - 2)
            num_bins = int(360 / constants.ANGLE_STEP)
            aggregated_elevations = np.full(num_bins, 0.0)

            with Pool(processes=num_processes) as pool:
                iterator = pool.imap_unordered(
                    process_tile_for_multiple_viewpoints, tasks
                )
                for tile_results in iterator:
                    for _, partial_skyline in tile_results:
                        aggregated_elevations = np.maximum(
                            aggregated_elevations, partial_skyline
                        )

            angles = np.linspace(0, 360, num_bins, endpoint=False)
            dem_mask_original = generate_skyline_mask_array(
                angles,
                aggregated_elevations,
                constants.PANORAMA_IMG_WIDTH,
                constants.PANORAMA_IMG_HEIGHT,
            )

            # --- CORE LOGIC CHANGE ---

            # 1. Calculate the yaw offset using the ORIGINAL real mask and ORIGINAL DEM mask
            yaw_offset = skyline_matcher.find_yaw_offset(dem_mask_original, real_mask)

            offset_results.append(
                {"filename": original_filename, "yaw_offset_deg": yaw_offset}
            )

            # 2. **NEW**: Modify the DEM mask to simulate FOV loss
            dem_mask_modified = modify_dem_mask_for_fov_loss(
                dem_mask=dem_mask_original,
                yaw_offset=yaw_offset,
                orientation=orientation_from_csv,
                manual_offset_deg=MANUAL_OFFSET_DEG,
                gap_width_degrees=GAP_WIDTH_DEG,
            )

            # 3. Rotate the real panorama using the calculated offset
            aligned_real_mask = rotate_panorama_by_orientation(real_mask, yaw_offset)

            # 4. Calculate similarity score using the ALIGNED real mask and the MODIFIED DEM mask
            similarity_score = calculate_skyline_similarity(
                dem_mask_modified, aligned_real_mask
            )

            tqdm.write(
                f"  -> For {original_filename}: Yaw Offset = {yaw_offset:.2f}°, "
                f"Post-Modification IoU = {similarity_score:.2%}"
            )

            # 5. Generate the comparison plot, passing the MODIFIED DEM mask for visualization
            save_path = output_dir / f"comparison_{image_number}.png"
            visualize_comparison(
                original_real_mask=real_mask,
                dem_mask=dem_mask_modified,  # Pass the modified mask here
                aligned_real_mask=aligned_real_mask,
                point_id=point_id,
                real_image_name=real_mask_name,
                similarity_score=similarity_score,
                yaw_offset=yaw_offset,
                save_path=save_path,
            )
            # You might want to update visualize_comparison to add a note in the title
            # like "(Modified for FOV Loss)" to avoid confusion.

        print("\nSaving calculated yaw offsets...")
        offsets_df = pd.DataFrame(offset_results)
        offsets_save_path = selected_dir / "yaw_offsets.csv"
        offsets_df.to_csv(offsets_save_path, index=False, float_format="%.4f")
        print(f"Yaw offsets saved to: {offsets_save_path}")

        print("\nBatch validation complete.")
        return True
    except Exception as e:
        print(f"An error occurred during batch validation: {e}")
        import traceback

        traceback.print_exc()
        return False


# TODO: For all work in the future where we refered to binary panorama we need to now refer to binary_panorama_aligned!!!
def run_panorama_alignment():
    """
    Stage 14: Rotates all binary panoramas in a dataset according to their
    pre-calculated yaw offsets and saves them to a new directory.
    """
    print_stage_header(14, "PANORAMA YAW ALIGNMENT")
    try:
        #  1. Select the dataset to process
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "yaw_offsets.csv").is_file()
            ]
        )
        if not dataset_dirs:
            print(
                "No datasets with a 'yaw_offsets.csv' file found. Please run Stage 13 first."
            )
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to align:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")
            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print("Invalid input.")
            except ValueError:
                print("Invalid input.")

        #  2. Load offsets and prepare directories
        offsets_csv_path = selected_dir / "yaw_offsets.csv"
        offsets_df = pd.read_csv(offsets_csv_path)
        offset_lookup = {
            row.filename: row.yaw_offset_deg for row in offsets_df.itertuples()
        }

        input_dir = selected_dir / "binary_panorama"
        output_dir = selected_dir / "binary_panorama_aligned"
        output_dir.mkdir(exist_ok=True)

        image_paths = sorted(list(input_dir.glob("*.png")))
        print(f"\nFound {len(image_paths)} panoramas to align. Saving to: {output_dir}")

        #  3. Loop, rotate, and save each image
        for img_path in tqdm(image_paths, desc="Aligning Panoramas"):
            # The key in the CSV is the original filename (e.g., '0801.jpg')
            image_number = re.search(r"(\d{4})", img_path.name).group(1)
            lookup_key = f"{image_number}.jpg"

            if lookup_key in offset_lookup:
                yaw_offset = offset_lookup[lookup_key]

                # Load the image with OpenCV
                image_to_rotate = cv2.imread(str(img_path))

                # Apply the rotation
                aligned_image = rotate_panorama_by_orientation(
                    image_to_rotate, yaw_offset
                )

                # Save to the new directory
                save_path = output_dir / img_path.name
                cv2.imwrite(str(save_path), aligned_image)

        print("\Panorama alignment complete.")
        return True

    except Exception as e:
        print(f"An error occurred during panorama alignment: {e}")
        import traceback

        traceback.print_exc()
        return False


# ! Warning: This method will take an enormous amount of time to run. If we want to have a finer grid than 25 meters, a good method now would be to shift the start and end point by 12,5 meters, and then run another 25 meter grid calculation.


# ! Important though! if we do the above, we need to make sure that we dont overwrite the old database, but that we instead concatenate to it!
def run_lidar_skyline_generation():
    """
    Stage 15: Generate a LiDAR skyline database with a correct, memory-safe,
    and resilient architecture.
    """
    print_stage_header(15, "LIDAR SKYLINE MASK GENERATION (MEMORY-SAFE)")

    try:
        MAX_CONCURRENT_LAZ_FILES = (
            10  # Cap the number of LAZ files processed in parallel
        )

        project_root = Path(__file__).resolve().parent
        db_dir = project_root / "databases" / "dem_database"
        db_dir.mkdir(parents=True, exist_ok=True)
        elevations_path = db_dir / "skyline_elevations.npz"
        final_masks_path = db_dir / "skyline_masks.npz"

        print_separator("PART A: Aggregating Elevation Data")

        # 1. Load ONLY the list of completed IDs to resume
        processed_point_ids = set()
        if elevations_path.exists():
            with np.load(elevations_path) as existing_data:
                processed_point_ids = set(existing_data.keys())
            print(f"Loaded {len(processed_point_ids)} existing points to resume.")

        # 2. Map LAZ files to viewpoints that STILL NEED to be processed
        grid_df = create_and_save_grid(
            start_coords=(611597.725, 6458055.924),
            end_coords=(614547.264, 6463806.450),
            spacing_meters=25.0,
        )
        all_coords = list(zip(grid_df["easting"], grid_df["northing"]))
        all_laz_files = find_laz_files()

        print("\nMapping LAZ files to remaining viewpoints")
        tile_to_viewpoints_map = {str(f): [] for f in all_laz_files}
        for x, y in tqdm(all_coords, desc="Planning tasks"):
            point_id = f"dem_vp_{x:.0f}_{y:.0f}"
            if point_id in processed_point_ids:
                continue
            viewpoint_coords = (x, y, 1.5)
            for laz_file in all_laz_files:
                if check_file_relevance_from_json(laz_file, viewpoint_coords):
                    tile_to_viewpoints_map[str(laz_file)].append(
                        (point_id, viewpoint_coords)
                    )

        relevant_laz_files = [
            Path(p) for p, vps in tile_to_viewpoints_map.items() if vps
        ]
        if not relevant_laz_files:
            print("\nAll viewpoints have already been processed.")
        else:
            # 3. Create batches OF LAZ FILES
            laz_batches = [
                relevant_laz_files[i : i + MAX_CONCURRENT_LAZ_FILES]
                for i in range(0, len(relevant_laz_files), MAX_CONCURRENT_LAZ_FILES)
            ]
            num_processes = max(1, cpu_count() - 6)

            # 4. Main batch processing loop
            for i, laz_batch in enumerate(laz_batches):
                print_separator(
                    f"Processing LAZ File Batch {i + 1} / {len(laz_batches)}"
                )

                # Create one coarse-grained task per LAZ file in the batch
                batch_tasks = [
                    (laz_path, tile_to_viewpoints_map[str(laz_path)])
                    for laz_path in laz_batch
                ]

                print(f"Processing {len(batch_tasks)} LAZ files in this batch...")

                batch_skyline_data = {}
                with Pool(processes=num_processes) as pool:
                    iterator = pool.imap_unordered(
                        process_tile_for_multiple_viewpoints, batch_tasks
                    )
                    with tqdm(total=len(batch_tasks), desc="Processing Tiles") as pbar:
                        for tile_results in iterator:
                            for point_id, partial_skyline in tile_results:
                                if point_id not in batch_skyline_data:
                                    batch_skyline_data[point_id] = partial_skyline
                                else:
                                    batch_skyline_data[point_id] = np.maximum(
                                        batch_skyline_data[point_id], partial_skyline
                                    )

                            # Update the progress bar after processing each tile
                            pbar.update(1)
                            # Display the number of unique points aggregated in THIS batch so far
                            pbar.set_postfix_str(
                                f"New Points in Batch: {len(batch_skyline_data)}"
                            )

                # 5. Merge and Save Incrementally
                if not batch_skyline_data:
                    print("No results generated in this batch.")
                    continue

                print(
                    f"\nMerging {len(batch_skyline_data)} new/updated points into database..."
                )
                existing_data = {}
                if elevations_path.exists():
                    with np.load(elevations_path, allow_pickle=True) as data:
                        existing_data = dict(data)

                for point_id, new_skyline in batch_skyline_data.items():
                    if point_id not in existing_data:
                        existing_data[point_id] = new_skyline
                    else:
                        existing_data[point_id] = np.maximum(
                            existing_data[point_id], new_skyline
                        )

                np.savez_compressed(elevations_path, **existing_data)
                print("Progress saved successfully.")
                gc.collect()

        print_separator("PART B: Generating Final Masks in Batches")

        if not elevations_path.exists():
            print("No elevation data found to process. Exiting.")
            return True

        # Create a dedicated directory for the mask batches
        mask_batches_dir = db_dir / "mask_batches"
        mask_batches_dir.mkdir(exist_ok=True)

        with np.load(elevations_path, allow_pickle=True) as elevation_data:
            all_point_ids = list(elevation_data.keys())

            # Find which batches still need to be created
            point_ids_to_process = all_point_ids
            print(
                f"Preparing to generate masks for {len(point_ids_to_process)} points..."
            )

            MASK_BATCH_SIZE = 500
            id_batches = [
                point_ids_to_process[i : i + MASK_BATCH_SIZE]
                for i in range(0, len(point_ids_to_process), MASK_BATCH_SIZE)
            ]

            for i, id_batch in enumerate(id_batches):
                batch_file_path = mask_batches_dir / f"mask_batch_{i:04d}.npz"
                # RESUME LOGIC: Skip batches that have already been saved
                if batch_file_path.exists():
                    continue

                print(f"\nProcessing mask batch {i + 1} / {len(id_batches)}")

                batch_masks = {}
                for point_id in tqdm(id_batch, desc="Generating Masks"):
                    elevations = elevation_data[point_id]
                    num_bins = len(elevations)
                    angles = np.linspace(0, 360, num_bins, endpoint=False)
                    batch_masks[point_id] = generate_skyline_mask_array(
                        angles,
                        elevations,
                        constants.PANORAMA_IMG_WIDTH,
                        constants.PANORAMA_IMG_HEIGHT,
                    )

                # Save only the current batch to its own file
                print(f"Saving {len(batch_masks)} masks to {batch_file_path.name}")
                np.savez_compressed(batch_file_path, **batch_masks)
                gc.collect()

        print_separator("PART C: Merging Batches into Final Database")

        batch_files = sorted(list(mask_batches_dir.glob("*.npz")))
        if not batch_files:
            print("No mask batches found to merge.")
            return True

        print(
            f"Merging {len(batch_files)} batch files into '{final_masks_path.name}'..."
        )

        with zipfile.ZipFile(
            final_masks_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf_out:
            for batch_file in tqdm(batch_files, desc="Merging Batches"):
                with zipfile.ZipFile(batch_file, "r") as zf_in:
                    for item in zf_in.namelist():
                        buffer = zf_in.read(item)
                        zf_out.writestr(item, buffer)

        print(
            f"\nLiDAR skyline mask generation complete. Final masks saved to '{final_masks_path}'."
        )
        return True

    except Exception as e:
        print(f"An error occurred during LiDAR processing: {e}")
        import traceback

        traceback.print_exc()
        return False


# TODO 1: Read the DEM skyline vectors and remove some degrees at the pixels (offset + the calculated shift)/5120 from the fourier IoU correction
# 2: Now that they are rotated to the original panoramas we know that the information loss is at 1/4:th plus and 3/4:th of the total width
# Now we remove some degree deg around these points and rotate the reapply the fourier matching between the panormas


# TODO: Maybe one thing we want to do, since the panoramas we have lose some information in the borders where the two camera images connect, perhaps before rotating the panoramas in north normalized direction, just like the dems are, we keep track of how much they need to be rotated, and then where the edges between the front and back facing images are, and cut out a couple of percent HFOV from the DEM simulated image there as well. This is because the DEM image has perfect 360deg coverage while our panoramas has maybe 2x175deg or something similar in practice, even though it should be 2x180deg.
def run_dem_skyline_encoding():
    """
    Stage 16: Load a DEM skyline mask database and create all fingerprint types
    in a memory-safe way.
    """
    print_stage_header(
        16,
        "DEM SKYLINE ENCODING",
        "Generates fingerprints from the DEM database in a streaming fashion.",
    )

    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"
        mask_db_path = db_root_dir / "dem_database" / "skyline_masks.npz"

        if not mask_db_path.is_file():
            print(f" DEM skyline mask database not found at '{mask_db_path}'.")
            print("   Please run Stage 14 first.")
            return False

        print(f"Opening DEM skyline database from {mask_db_path}...")

        # Use lazy-loading to open the database without loading it all into RAM
        with np.load(mask_db_path) as skyline_mask_data:
            print(
                f"Found {len(skyline_mask_data.files)} simulated skylines to process."
            )

            dataset_db_dir = db_root_dir / "dem_database"

            # Fourier encoding
            for strategy in ["global", "windowed"]:
                # Pass the lazy-loaded data object to the new helper function
                fourier_fps = (
                    fourier_skyline_encoder.create_fingerprints_from_lazy_loader(
                        skyline_mask_data, strategy=strategy
                    )
                )
                if fourier_fps:
                    output_path = (
                        dataset_db_dir
                        / f"fourier_{strategy}_fingerprints"
                        / "fingerprints.npz"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output_path, **fourier_fps)
                    print(
                        f"Saved DEM Fourier ({strategy}) fingerprints to {output_path}"
                    )

            # CNN encoding
            cnn_fps = cnn_skyline_encoder.create_cnn_fingerprints_from_lazy_loader(
                skyline_mask_data,
                pool_target=constants.POOL_TARGET,  # Using a 2x2 grid for spatial awareness
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


def run_database_visualization():
    """
    Stage 17: Compares a real panorama to the closest mask in the DEM database.
    """
    print_stage_header(
        17,
        "DATABASE VISUALIZATION",
        "Comparing a real panorama to its closest DEM match.",
    )
    try:
        project_root = Path(__file__).parent
        db_path = project_root / "databases" / "dem_database" / "skyline_masks.npz"

        if not db_path.exists():
            print(f"ERROR: Final database not found at '{db_path}'.")
            return False

        #  1. Select the dataset interactively
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
            print("No datasets with GPS files found to select a coordinate from.")
            return False

        selected_dir = None
        while True:
            print("\nPlease select a dataset to provide a coordinate:")
            for i, dir_path in enumerate(dataset_dirs):
                print(f"  [{i}]: {dir_path.name}")
            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(dataset_dirs):
                    selected_dir = dataset_dirs[choice_idx]
                    break
                else:
                    print("Invalid input.")
            except ValueError:
                print("Invalid input.")

        #  2. Select the specific data point by its 'n' value
        gps_csv_path = selected_dir / "gps_interpolated_with_orientation.csv"
        gps_df = pd.read_csv(gps_csv_path)

        selected_row = None
        while True:
            print("\nPlease select a data point to find the closest match for:")
            print(gps_df.head(10).to_string())
            choice_str = input("Enter the value from the 'n' column: ")
            try:
                choice_n_val = int(choice_str)
                # Find the row where the 'n' column matches the input value
                selected_row = gps_df[gps_df["n"] == choice_n_val]

                if not selected_row.empty:
                    if len(selected_row) > 1:
                        print(
                            f"Warning: Found multiple entries for n={choice_n_val}. Using the first one."
                        )
                    break  # Exit loop if a match is found
                else:
                    print(
                        f"Invalid input. Value '{choice_n_val}' not found in the 'n' column."
                    )
            except ValueError:
                print("Invalid input. Please enter a number.")

        #  3. Extract data and find corresponding files
        query_coord = np.array(
            [selected_row["easting"].iloc[0], selected_row["northing"].iloc[0]]
        )
        original_filename = selected_row["filename"].iloc[0]

        image_number = Path(original_filename).stem
        real_mask_name = f"binary_panorama_{image_number}.png"
        real_mask_path = selected_dir / "binary_panorama_aligned" / real_mask_name

        if not real_mask_path.exists():
            print(
                f"ERROR: Corresponding binary panorama not found at '{real_mask_path}'"
            )
            return False

        real_mask = np.array(Image.open(real_mask_path))

        #  4. Search the database for the closest point
        print("\nSearching database for the closest grid point...")
        with np.load(db_path) as data:
            all_point_ids = data.files
            db_coords = np.array(
                [
                    [int(pid.split("_")[2]), int(pid.split("_")[3])]
                    for pid in all_point_ids
                ]
            )

            distances = np.sum((db_coords - query_coord) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            closest_point_id = all_point_ids[closest_idx]

            print(f"Query Coordinate: ({query_coord[0]:.1f}, {query_coord[1]:.1f})")
            print(f"Closest Found ID: {closest_point_id}")

            dem_mask_array = data[closest_point_id]

            #  5. Visualize both images
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            fig.suptitle(f"Comparison for Query '{original_filename}'", fontsize=14)

            ax1.imshow(real_mask, cmap="gray", aspect="auto")
            ax1.set_title(f"Real Segmented Panorama ({real_mask_name})")
            ax1.set_xlabel("Azimuth")
            ax1.set_ylabel("Elevation")

            ax2.imshow(dem_mask_array, cmap="gray", aspect="auto")
            ax2.set_title(f"Closest DEM Skyline Mask ({closest_point_id})")
            ax2.set_xlabel("Azimuth")
            ax2.set_ylabel("Elevation")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            output_dir = project_root / "output_plots"
            output_dir.mkdir(exist_ok=True)
            save_path = (
                output_dir
                / "real_closest_database_point_comparison"
                / f"db_comparison_{image_number}.png"
            )
            plt.savefig(save_path)
            plt.close()
            print(f"Comparison visualization saved to: {save_path}")

        return True

    except Exception as e:
        print(f"An error occurred during database visualization: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_skyline_fourier_encoder():
    """
    Stage 18: Select a dataset and create fingerprint databases in the project root
    under a new per-dataset directory structure.
    """
    print_stage_header(
        18,
        "SKYLINE FINGERPRINTING",
        "Generating 'global' and 'windowed' fingerprint databases.",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "binary_panorama_aligned").is_dir()
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

        input_binary_dir = selected_dir / "binary_panorama_aligned"
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
    Stage 19: Perform a self-matching analysis on a fingerprint database,
    showing detailed per-window scores for windowed strategies.
    """
    print_stage_header(
        19,
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
            num_windows = constants.NUM_WINDOWS
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
    Stage 20: Select a dataset and create CNN-based fingerprints for its binary panoramas.
    """
    print_stage_header(
        20,
        "CNN SKYLINE ENCODING",
        "Generating rich feature vectors using a pre-trained CNN.",
    )

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "binary_panorama_aligned").is_dir()
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

        input_binary_dir = selected_dir / "binary_panorama_aligned"
        dataset_name = selected_dir.name

        # Define the new output directory structure
        project_root = Path(__file__).parent
        output_dir = project_root / "databases" / dataset_name / "cnn_fingerprints"
        output_path = output_dir / "fingerprints.npz"

        exit_code = cnn_skyline_encoder.create_cnn_fingerprints_for_directory(
            input_dir=input_binary_dir,
            output_filepath=output_path,
            pool_target=constants.POOL_TARGET,  # Must also be (2, 2)
        )

        return exit_code == 0

    except Exception as e:
        print(f"An error occurred during CNN encoding: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_cnn_matching_analysis():
    """
    Stage 21: Perform a self-matching analysis on a CNN-based fingerprint
    database to validate the encoding method.
    """
    print_stage_header(
        21,
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
            fingerprint_path=selected_fp_path,
            num_samples=10,
            top_n=5,
            num_windows=constants.NUM_WINDOWS,
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


def run_fourier_cross_matching_analysis():
    """
    Stage 22: Match real image Fourier fingerprints against the simulated DEM database.
    """
    print_stage_header(22, "FOURIER CROSS-MATCHING ANALYSIS (REAL VS. DEM)")
    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"

        available_query_dbs = sorted(list(db_root_dir.glob("dataset*/*fourier*/*.npz")))
        if not available_query_dbs:
            print("No real-image Fourier fingerprint databases found.")
            return False

        selected_query_path = None
        while True:
            print("\nPlease select a query fingerprint file (real images):")
            for i, file_path in enumerate(available_query_dbs):
                print(f"  [{i}]: {file_path.relative_to(project_root)}")
            choice_str = input("Enter number: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_query_dbs):
                    selected_query_path = available_query_dbs[choice_idx]
                    break
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

        strategy = "windowed" if "windowed" in selected_query_path.name else "global"
        dem_fp_db_path = (
            db_root_dir
            / "dem_database"
            / f"fourier_{strategy}_fingerprints"
            / "fingerprints.npz"
        )
        dem_mask_db_path = (
            db_root_dir / "dem_database" / "skyline_masks.npz"
        )  # Path to the DEM masks

        dataset_name = selected_query_path.parent.parent.name
        gps_csv_path = (
            Path(constants.IMAGES_DIR)
            / dataset_name
            / "gps_interpolated_with_orientation.csv"
        )

        print(gps_csv_path)

        results, mean_error = skyline_matcher.run_cross_matching(
            query_fp_path=selected_query_path,
            dem_fp_path=dem_fp_db_path,
            gps_csv_path=gps_csv_path,
        )

        if results:
            print_separator(
                f"Cross-Matching Results (Mean Top-1 Error: {mean_error:.2f} m)"
            )

            output_plot_dir = (
                project_root / "output_plots" / f"fourier_cross_match_{dataset_name}"
            )

            for result in results:
                print(f"\n  Query: {result['query']}")
                for match in result["top_matches"]:
                    print(
                        f"    -> Match: {match['match_key']:<30} | Score: {match['score']:.4f} | Error: {match['error_m']:.2f} m"
                    )

                # Find the query image path
                query_image_path = (
                    Path(constants.IMAGES_DIR)
                    / dataset_name
                    / "binary_panorama_aligned"
                    / f"{result['query']}.png"
                )
                if query_image_path.exists():
                    plot_save_path = output_plot_dir / f"{result['query']}_matches.png"
                    create_cross_match_plot(
                        query_image_path=query_image_path,
                        top_matches=result["top_matches"],
                        dem_mask_db_path=dem_mask_db_path,
                        save_path=plot_save_path,
                    )
            print(f"\nVisualizations saved to: {output_plot_dir}")

        return True
    except Exception as e:
        print(f"An error occurred during cross-matching: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_cnn_cross_matching_analysis():
    """
    Stage 23: Match real image CNN fingerprints against the simulated DEM database.
    """
    print_stage_header(23, "CNN CROSS-MATCHING ANALYSIS (REAL VS. DEM)")
    try:
        project_root = Path(__file__).parent
        db_root_dir = project_root / "databases"

        #  Select the REAL image database to use as queries
        available_query_dbs = sorted(
            list(db_root_dir.glob("dataset*/cnn_fingerprints/*.npz"))
        )
        if not available_query_dbs:
            print("No real-image CNN fingerprint databases found.")
            return False

        selected_query_path = None
        while True:
            print("\nPlease select a query fingerprint file (real images):")
            for i, file_path in enumerate(available_query_dbs):
                print(f"  [{i}]: {file_path.relative_to(project_root)}")
            choice_str = input("Enter number: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_query_dbs):
                    selected_query_path = available_query_dbs[choice_idx]
                    break
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

        dem_fp_db_path = (
            db_root_dir / "dem_database" / "cnn_fingerprints" / "fingerprints.npz"
        )
        dem_mask_db_path = db_root_dir / "dem_database" / "skyline_masks.npz"

        dataset_name = selected_query_path.parent.parent.name
        gps_csv_path = (
            Path(constants.IMAGES_DIR)
            / dataset_name
            / "gps_interpolated_with_orientation.csv"
        )

        results, mean_error = skyline_matcher.run_cross_matching(
            query_fp_path=selected_query_path,
            dem_fp_path=dem_fp_db_path,
            gps_csv_path=gps_csv_path,
            num_samples=20,
            top_n=5,
            num_windows=constants.NUM_WINDOWS,
        )

        if results:
            print_separator(
                f"Cross-Matching Results (Mean Top-1 Error: {mean_error:.2f} m)"
            )

            output_plot_dir = (
                project_root / "output_plots" / f"cnn_cross_match_{dataset_name}"
            )

            for result in results:
                print(f"\n  Query: {result['query']}")
                for match in result["top_matches"]:
                    print(
                        f"    -> Match: {match['match_key']:<30} | Score: {match['score']:.4f} | Error: {match['error_m']:.2f} m"
                    )

                # Find the query image path
                query_image_path = (
                    Path(constants.IMAGES_DIR)
                    / dataset_name
                    / "binary_panorama_aligned"
                    / f"{result['query']}.png"
                )
                if query_image_path.exists():
                    plot_save_path = output_plot_dir / f"{result['query']}_matches.png"
                    create_cross_match_plot(
                        query_image_path=query_image_path,
                        top_matches=result["top_matches"],
                        dem_mask_db_path=dem_mask_db_path,
                        save_path=plot_save_path,
                    )
            print(f"\nVisualizations saved to: {output_plot_dir}")
        return True
    except Exception as e:
        print(f"An error occurred during cross-matching: {e}")
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
        "3b": ("Run K-Fold Cross-Validation", run_roll_angle_cross_validation),
        4: ("Evaluate Roll Angle Model", run_roll_model_evaluation),
        5: ("Interactive Roll Prediction", run_interactive_roll_prediction),
        6: ("Visualize CNN Features", run_feature_visualization),
        7: ("Undistort Fisheye", run_fisheye_undistortion),
        8: ("Process GPS/Orientation", run_gps_orientation_processing),
        9: ("Stitch Panoramas", run_panorama_stitching),
        10: ("Segment Skylines", run_skyline_segmentation),
        11: (
            "Build COLMAP Databases",
            lambda: build_colmap_databases(args.rebuild_colmap),
        ),
        12: (
            "Analyze Retrieval Accuracy",
            lambda: analyze_retrieval_accuracy(args.sample_rate),
        ),
        13: (
            "Batch Point Validation same Coordinates with DEM and Panorama",
            run_batch_validation,
        ),
        14: ("Apply Panorama Alignment", run_panorama_alignment),
        15: ("Generate DEM Skyline Masks", run_lidar_skyline_generation),
        16: ("Encode DEM Skylines", run_dem_skyline_encoding),
        17: ("Visualize Database Mask", run_database_visualization),
        18: ("Encode REAL Skylines (Fourier)", run_skyline_fourier_encoder),
        19: ("Encode REAL Skylines (CNN)", run_cnn_encoding),
        20: ("Fourier Database Self-Matching Analysis", run_matching_analysis),
        21: ("CNN Database Self-Matching Analysis", run_cnn_matching_analysis),
        22: (
            "Fourier Cross-Matching (Real vs. DEM)",
            run_fourier_cross_matching_analysis,
        ),
        23: ("CNN Cross-Matching (Real vs. DEM)", run_cnn_cross_matching_analysis),
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
