import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

from utils import split_stacked_images 

# Use only one of these two, either polar mist undistortion from their estimated campera parameters our ours from the given csv from camera manifacturer
from utils.fisheye_undistorter import FisheyeUndistorter
# from utils.use_polarmist_undistorter import main as fisheye_undistorter_main

from utils.create_panoramas import build_panoramas

from utils import orientation_path_from_gps
from utils.utility import get_original_numbered_file_name

from image_retrieval.colmap_naive_image_retrieval import ColmapImageRetriever
from image_retrieval.colmap_naive_image_retrieval import retrieve_top_matches

#from utils.skyline_segmenter import segment_skyline_from_panoramas, get_sam_predictor
from utils.segformer_segmenter import SemanticSegmenter

from utils.geospatial_processing import create_dsm_from_laz, generate_simulated_skylines

import constants

def print_separator(title=""):
    """Print a visual separator for pipeline stages."""
    print("\n" + "="*60)
    if title:
        print(f" {title} ")
        print("="*60)
    else:
        print("="*60)


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
            "rotation_cam2": -9.85
        }
    else:
        # Default parameters for all other datasets
        return {
            "vertical_offset": 0,
            "horizon_bend_cam1": 15,
            "horizon_bend_cam2": -15,
            "rotation_cam1": 4.35,
            "rotation_cam2": -4.65
        }

def run_image_splitting():
    """
    Stage 1: Split stacked images into individual camera images.
    """
    print_stage_header(1, "IMAGE SPLITTING", 
                    "Splitting stacked images into cam1 and cam2 images")
    
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


def run_fisheye_undistortion():
    """
    Stage 2: Undistort fisheye images for all datasets.
    - Finds each dataset folder in the images directory.
    - Applies roll and horizon bend correction and a symmetrical vertical pitch correction to both cameras.
    """
    print_stage_header(2, "FISHEYE UNDISTORTION",
                    "Applying camera-specific alignment parameters to a selected dataset.")

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find and sort datasets that have been processed by Stage 1
        dataset_dirs = sorted([d for d in datasets_root_dir.iterdir() if d.is_dir() and (d / "distorted_images_cam1").is_dir()])

        if not dataset_dirs:
            print("No processed dataset folders found (e.g., 'images/dataset1/distorted_images_cam1').")
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
                    print(f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}.")
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")

        print(f"\n--- Processing dataset: {selected_dir.name} ---")

        # Initialize the undistorter
        undistorter = FisheyeUndistorter(csv_path=constants.DISTORTION_PARAMS_CSV)

        # Get the specific parameters for this dataset
        params = get_undistortion_parameters(selected_dir)

        # Calculate pitch adjustment based on the retrieved parameters
        half_offset = params['vertical_offset'] / 2.0
        pitch_adjustment_angle = FisheyeUndistorter.calculate_pitch_correction_from_offset(
            half_offset,
            constants.OUTPUT_IMG_WIDTH,
            90
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
            rotation_degrees=params['rotation_cam1'],
            horizon_bend_correction=params['horizon_bend_cam1']
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
            rotation_degrees=params['rotation_cam2'],
            horizon_bend_correction=params['horizon_bend_cam2']
        )

        print(f"\n✅ Fisheye undistortion for '{selected_dir.name}' finished successfully!")
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
    print_stage_header(3, "GPS & ORIENTATION PROCESSING", 
                    "Calculates orientation from GPS data for a selected dataset.")
    
    try:
        INPUT_CSV_NAME = "gps_interpolated.csv"
        source_datasets_dir = Path(constants.IMAGES_DIR) / "original_datasets"
        
        available_datasets = sorted([
            d for d in source_datasets_dir.iterdir() if d.is_dir() and (d / INPUT_CSV_NAME).is_file()
        ])

        if not available_datasets:
            print(f"No source dataset folders found containing '{INPUT_CSV_NAME}'. Searched in: {source_datasets_dir}")
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
                    print(f"\n❌ Invalid input. Please enter a number between 0 and {len(available_datasets) - 1}.")
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
        df['Orientation'] = orientation_path_from_gps.calculate_orientation(df)

        print(f"Saving data with orientation to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False, float_format='%.3f')

        orientation_path_from_gps.plot_trajectory(
            df, title_suffix=f"from {selected_source_dir.name}", save_path=output_plot_path
        )

        print(f"\n✅ GPS orientation processing for '{selected_source_dir.name}' complete.")
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
    Stage 4: Stitch the four rectilinear views for each dataset into panoramas.
    """
    print_stage_header(4, "PANORAMA STITCHING", 
                    "Combining four views into a single panoramic image per timestamp for each dataset")
    
    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find datasets that have been processed by Stage 2
        dataset_dirs = [d for d in datasets_root_dir.iterdir() if d.is_dir() and (d / "undistorted_cam1_left").is_dir()]

        if not dataset_dirs:
            print("No undistorted dataset folders found (e.g., 'images/dataset1/undistorted_cam1_left').")
            print("Please run Stage 2 first.")
            return False

        all_successful = True
        for dataset_dir in dataset_dirs:
            print(f"\n--- Stitching panoramas for dataset: {dataset_dir.name} ---")

            orientation_csv_path = dataset_dir / "gps_interpolated_with_orientation.csv"
            if not orientation_csv_path.is_file():
                print(f"❌ Orientation CSV not found for {dataset_dir.name} at '{orientation_csv_path}'.")
                print("   Please run Stage 3 for this dataset first. Skipping panorama stitching.")
                all_successful = False
                continue

            # Define input and output paths for the current dataset
            cam1_left_dir = dataset_dir / "undistorted_cam1_left"
            cam1_right_dir = dataset_dir / "undistorted_cam1_right"
            cam2_left_dir = dataset_dir / "undistorted_cam2_left"
            cam2_right_dir = dataset_dir / "undistorted_cam2_right"
            output_panorama_dir = dataset_dir / "panoramas"

            exit_code = build_panoramas(
                cam1_left_dir=cam1_left_dir,
                cam1_right_dir=cam1_right_dir,
                cam2_left_dir=cam2_left_dir,
                cam2_right_dir=cam2_right_dir,
                output_dir=output_panorama_dir,
                orientation_csv_path=orientation_csv_path
            )
            
            if exit_code != 0:
                print(f"Panorama stitching failed for dataset: {dataset_dir.name}!")
                all_successful = False

        if all_successful:
            print("\nPanorama stitching completed successfully for all datasets!")
        else:
            print("\nPanorama stitching failed for one or more datasets.")
            
        return all_successful
            
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
#def run_skyline_segmentation():
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
    print_stage_header(5, "SKYLINE SEGMENTATION",
                    "Generating binary sky masks using segformer-b5 for a selected dataset")

    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        # Find and sort datasets that have been processed by Stage 4
        dataset_dirs = sorted([
            d for d in datasets_root_dir.iterdir() if d.is_dir() and (d / "panoramas").is_dir()
        ])

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
                    print(f"\n❌ Invalid input. Please enter a number between 0 and {len(dataset_dirs) - 1}.")
            except ValueError:
                print("\n❌ Invalid input. Please enter a number.")
        
        print(f"\n--- Segmenting skylines for selected dataset: {selected_dir.name} ---")

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
            output_dir_overlay=output_overlay_dir
        )
        
        if exit_code == 0:
            print(f"\n✅ Skyline segmentation completed successfully for {selected_dir.name}!")
            return True
        else:
            print(f"\n❌ Skyline segmentation failed for {selected_dir.name}!")
            return False

    except Exception as e:
        print(f"An error occurred during skyline segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

# TODO: This is just a placeholder initial method now that we are working with other images than the distorted ones
def build_colmap_databases(force_rebuild: bool = False):
    """
    Stage 6: Build COLMAP databases for image retrieval for each dataset.
    """
    print_stage_header(6, "BUILD COLMAP DATABASES",
                    "Building feature databases for each dataset")
    
    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = [d for d in datasets_root_dir.iterdir() if d.is_dir() and (d / "distorted_images_cam1").is_dir()]

        if not dataset_dirs:
            print("No dataset folders with distorted images found. Please run Stage 1 first.")
            return False
        
        all_successful = True
        for dataset_dir in dataset_dirs:
            print(f"\n--- Building COLMAP databases for dataset: {dataset_dir.name} ---")

            # NOTE: For undistorted images, we are using the 'left' views. 
            # This might need adjustment based on your specific retrieval strategy.
            configs = [
                {
                    "name": "Distorted Images",
                    "cam1_folder": dataset_dir / "distorted_images_cam1",
                    "cam2_folder": dataset_dir / "distorted_images_cam2",
                    "cam1_type": "distorted", "cam2_type": "distorted",
                    "cam1_model": "OPENCV_FISHEYE", "cam2_model": "OPENCV_FISHEYE"
                },
                {
                    "name": "Undistorted Images (Left Views)",
                    "cam1_folder": dataset_dir / "undistorted_cam1_left",
                    "cam2_folder": dataset_dir / "undistorted_cam2_left",
                    "cam1_type": "undistorted", "cam2_type": "undistorted",
                    "cam1_model": "SIMPLE_RADIAL", "cam2_model": "SIMPLE_RADIAL"
                }
            ]
            
            for config in configs:
                print(f"\nBuilding for {config['name']}...")
                
                retriever = ColmapImageRetriever(
                    cam1_folder=config["cam1_folder"],
                    cam2_folder=config["cam2_folder"],
                    cam1_camera_model=config["cam1_model"],
                    cam2_camera_model=config["cam2_model"],
                    cam1_type=config["cam1_type"],
                    cam2_type=config["cam2_type"]
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
    print_stage_header(7, "RETRIEVAL ACCURACY ANALYSIS",
                    "Analyzing location errors for each dataset")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retrieval_accuracy_results_{timestamp}.txt"
    
    results_output = []
    results_output.append("="*80)
    results_output.append("RETRIEVAL ACCURACY ANALYSIS RESULTS")
    results_output.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results_output.append(f"Sample rate: Every {sample_rate}th image")
    
    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        dataset_dirs = sorted([d for d in datasets_root_dir.iterdir() if d.is_dir() and (d / "distorted_images_cam1").is_dir()])

        if not dataset_dirs:
            print("No dataset folders found. Please run Stage 1 first.")
            return False

        for dataset_dir in dataset_dirs:
            results_output.append("\n" + "="*80)
            results_output.append(f"DATASET: {dataset_dir.name}")
            results_output.append("="*80)

            # Load the specific orientation CSV for this dataset
            orientation_csv_path = dataset_dir / "gps_interpolated_with_orientation.csv"
            if not orientation_csv_path.is_file():
                print(f"❌ Orientation CSV not found for {dataset_dir.name}. Skipping analysis.")
                results_output.append("Orientation CSV not found. Skipping analysis for this dataset.")
                continue

            print(f"Loading GPS data for {dataset_dir.name} from {orientation_csv_path}...")
            gps_df = pd.read_csv(orientation_csv_path)
            gps_lookup = {row['filename']: (row['easting'], row['northing'])
                        for _, row in gps_df.iterrows()}
            results_output.append(f"Loaded GPS data for {len(gps_lookup)} images.")

            # NOTE: For undistorted images, we analyze the 'left' views.
            analysis_configs = [
                {
                    "name": "Distorted Images Analysis",
                    "cam1_folder": dataset_dir / "distorted_images_cam1",
                    "cam2_folder": dataset_dir / "distorted_images_cam2",
                    "cam1_type": "distorted", "cam2_type": "distorted",
                    "cam1_model": "OPENCV_FISHEYE", "cam2_model": "OPENCV_FISHEYE",
                    "query_type": "distorted"
                },
                {
                    "name": "Undistorted Images Analysis (Left Views)",
                    "cam1_folder": dataset_dir / "undistorted_cam1_left",
                    "cam2_folder": dataset_dir / "undistorted_cam2_left",
                    "cam1_type": "undistorted", "cam2_type": "undistorted",
                    "cam1_model": "SIMPLE_RADIAL", "cam2_model": "SIMPLE_RADIAL",
                    "query_type": "undistorted"
                }
            ]
            
            for config in analysis_configs:
                print(f"\n--- {config['name']} for {dataset_dir.name} ---")
                results_output.append(f"\n--- {config['name']} ---")
                
                for cam_name, folder in [("cam1", config["cam1_folder"]), 
                                        ("cam2", config["cam2_folder"])]:
                    
                    if not folder.is_dir():
                        print(f"Folder not found, skipping: {folder}")
                        continue

                    print(f"\nAnalyzing {cam_name}...")
                    results_output.append(f"\nAnalyzing {cam_name}...")
                    
                    all_images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")),
                                        key=lambda x: int(get_original_numbered_file_name(x.name).split('.')[0]))
                    
                    if not all_images:
                        print(f"No images found in {folder}")
                        results_output.append(f"No images found in {folder}")
                        continue
                    
                    sample_images = all_images[::sample_rate]
                    print(f"Found {len(all_images)} images, sampling {len(sample_images)}")
                    
                    all_errors, missing_gps, no_matches, processed = [], 0, 0, 0
                    
                    for i, query_image in enumerate(sample_images):
                        original_filename = get_original_numbered_file_name(query_image.name)
                        if original_filename not in gps_lookup:
                            missing_gps += 1
                            continue
                        
                        query_easting, query_northing = gps_lookup[original_filename]
                        
                        matches = retrieve_top_matches(
                            query_image_path=query_image,
                            cam1_folder=config["cam1_folder"], cam2_folder=config["cam2_folder"],
                            cam1_camera_model=config["cam1_model"], cam2_camera_model=config["cam2_model"],
                            cam1_type=config["cam1_type"], cam2_type=config["cam2_type"],
                            query_image_type=config["query_type"], top_n=5
                        )
                        
                        if not matches:
                            no_matches += 1
                            continue
                        
                        processed += 1
                        for match_name, _ in matches:
                            original_match_filename = get_original_numbered_file_name(match_name)
                            if original_match_filename in gps_lookup:
                                match_easting, match_northing = gps_lookup[original_match_filename]
                                error = np.sqrt((query_easting - match_easting)**2 + (query_northing - match_northing)**2)
                                all_errors.append(error)
                    
                    # ... (results printing and logging) ...
                    print(f"  Images with matches: {processed}/{len(sample_images)}")
                    results_output.append(f"  Images with matches: {processed}/{len(sample_images)}")
                    if all_errors:
                        mean_error = np.mean(all_errors)
                        print(f"  Mean Error: {mean_error:.2f} meters")
                        results_output.append(f"  Mean Error: {mean_error:.2f} meters")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(results_output))
        
        print(f"\nResults saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during retrieval analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def run_lidar_skyline_generation():
    """
    Stage 8: Create a DSM from LiDAR data and generate simulated skylines.
    NOTE: This is assumed to be a global process using a single LiDAR dataset
    and grid file, so it does not loop through the individual image datasets.
    """
    print_stage_header(8, "LIDAR SKYLINE GENERATION",
                    "Creating DSM and simulating views from a grid")
    
    dsm_path = constants.GEOSPATIAL_OUTPUT_DIR / "DSM_merged.tif"
    sim_dir = constants.GEOSPATIAL_OUTPUT_DIR / "simulated_skylines"
    
    constants.GEOSPATIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\nPart 1: Creating DSM from LAZ files")
        dsm_success = create_dsm_from_laz(
            laz_dir=constants.LIDAR_DATA_DIR,
            output_dsm_path=dsm_path,
            resolution=1.0
        )
        if not dsm_success:
            print("DSM creation failed. Aborting stage.")
            return False
        
        print("\nPart 2: Generating simulated skylines from DSM")
        skyline_success = generate_simulated_skylines(
            dsm_path=dsm_path,
            grid_path=constants.OBSERVER_GRID_PATH,
            output_dir=sim_dir
        )
        if not skyline_success:
            print("Simulated skyline generation failed.")
            return False
        
        print("\nLiDAR skyline generation completed successfully!")
        return True

    except Exception as e:
        print(f"An error occurred during LiDAR processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Fisheye image processing pipeline')
    parser.add_argument('--skip-stages', nargs='+', type=int, default=[], 
                    help='Stages to skip (e.g., --skip-stages 1 2)')
    parser.add_argument('--only-stages', nargs='+', type=int, default=[], 
                    help='Only run specific stages (e.g., --only-stages 3 4)')
    parser.add_argument('--rebuild-colmap', action='store_true',
                    help='Force rebuild of COLMAP databases')
    parser.add_argument('--sample-rate', type=int, default=10,
                    help='Sample rate for analysis (default: every 10th image)')
    
    args = parser.parse_args()
    # Define all pipeline stages
    stages = {
        1: ("Split Images", run_image_splitting),
        2: ("Undistort Fisheye", run_fisheye_undistortion),
        3: ("Process GPS/Orientation", run_gps_orientation_processing),
        4: ("Stitch Panoramas", run_panorama_stitching),
        5: ("Segment Skylines", run_skyline_segmentation),
        6: ("Build COLMAP Databases", lambda: build_colmap_databases(args.rebuild_colmap)),
        7: ("Analyze Retrieval Accuracy", lambda: analyze_retrieval_accuracy(args.sample_rate)),
        8: ("Generate LiDAR Skylines", run_lidar_skyline_generation),
        #9: ("Match Skylines", run_skyline_matching) # TODO: Implement this
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