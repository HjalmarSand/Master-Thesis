import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

from utils import split_stacked_images 

# Use only one of these two, either polar mist undistortion from their estimated campera parameters our ours from the given csv from camera manifacturer
from utils.fisheye_undistorter import main as fisheye_undistorter_main
# from utils.use_polarmist_undistorter import main as fisheye_undistorter_main

from utils import orientation_path_from_gps
from utils.utility import get_original_numbered_file_name

from image_retrieval.colmap_naive_image_retrieval import ColmapImageRetriever
from image_retrieval.colmap_naive_image_retrieval import retrieve_top_matches

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


def run_image_splitting():
    """
    Stage 1: Split stacked images into individual camera images.
    """
    print_stage_header(1, "IMAGE SPLITTING", 
                    "Splitting stacked images into cam1 and cam2 images")
    
    try:
        exit_code = split_stacked_images.main()
        
        if exit_code == 0 or exit_code is None:  # Handle None return
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
    Stage 2: Undistort fisheye images.
    """
    print_stage_header(2, "FISHEYE UNDISTORTION", 
                    "Correcting fisheye distortion in camera images")
    
    try:
        exit_code = fisheye_undistorter_main()
        
        if exit_code == 0:
            print("Fisheye undistortion completed successfully!")
            return True
        else:
            print("Fisheye undistortion failed!")
            return False
            
    except Exception as e:
        print(f"Error during fisheye undistortion: {e}")
        return False


def run_gps_orientation_processing():
    """
    Stage 3: Process GPS and orientation data.
    """
    print_stage_header(3, "GPS & ORIENTATION PROCESSING", 
                    "Processing GPS coordinates and orientation data")
    
    try:
        exit_code = orientation_path_from_gps.main()
        
        if exit_code == 0:
            print("GPS orientation processing completed successfully!")
            return True
        else:
            print("GPS orientation processing failed!")
            return False
            
    except Exception as e:
        print(f"Error during GPS orientation processing: {e}")
        return False
    

def build_colmap_databases(force_rebuild: bool = False):
    """
    Stage 4: Build COLMAP databases for image retrieval
    Builds databases for all combinations: distorted/undistorted x cam1/cam2
    """
    print_stage_header(4, "BUILD COLMAP DATABASES", 
                    "Building feature databases for image retrieval")
    
    try:
        # Configuration for all database combinations
        configs = [
            {
                "name": "Distorted Images",
                "cam1_folder": constants.CAM1_DISTORTED_FOLDER,
                "cam2_folder": constants.CAM2_DISTORTED_FOLDER,
                "cam1_type": "distorted",
                "cam2_type": "distorted",
                "cam1_model": "OPENCV_FISHEYE",
                "cam2_model": "OPENCV_FISHEYE"
            },
            {
                "name": "Undistorted Images",
                "cam1_folder": constants.CAM1_UNDISTORTED_FOLDER,
                "cam2_folder": constants.CAM2_UNDISTORTED_FOLDER,
                "cam1_type": "undistorted",
                "cam2_type": "undistorted",
                "cam1_model": "SIMPLE_RADIAL",
                "cam2_model": "SIMPLE_RADIAL"
            }
        ]
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Building databases for {config['name']}")
            print(f"{'='*60}")
            
            # Create retriever instance
            retriever = ColmapImageRetriever(
                cam1_folder=config["cam1_folder"],
                cam2_folder=config["cam2_folder"],
                cam1_camera_model=config["cam1_model"],
                cam2_camera_model=config["cam2_model"],
                cam1_type=config["cam1_type"],
                cam2_type=config["cam2_type"]
            )
            
            # Initialize databases
            retriever.initialize_databases(force_rebuild=force_rebuild)
            
        print("\nAll COLMAP databases built successfully")
        return True
        
    except Exception as e:
        print(f"Error building COLMAP databases: {e}")
        return False


def analyze_retrieval_accuracy(sample_rate: int = 10):
    """
    Stage 5: Analyze image retrieval accuracy by comparing GPS locations
    
    Args:
        sample_rate: Sample every Nth image (default 10)
    """
    print_stage_header(5, "RETRIEVAL ACCURACY ANALYSIS", 
                    "Analyzing location errors between queries and matches")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retrieval_accuracy_results_{timestamp}.txt"
    
    # List to store all results for the output file
    results_output = []
    results_output.append("="*60)
    results_output.append("RETRIEVAL ACCURACY ANALYSIS RESULTS")
    results_output.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results_output.append(f"Sample rate: Every {sample_rate}th image")
    results_output.append("="*60)
    
    try:
        # Load GPS data
        gps_df = pd.read_csv(constants.GPS_LOCATION_AND_ORIENTATION)

        # Create a lookup dictionary for faster access
        gps_lookup = {row['filename']: (row['easting'], row['northing']) 
                    for _, row in gps_df.iterrows()}
        
        print(f"Loaded GPS data for {len(gps_lookup)} images")
        results_output.append(f"\nLoaded GPS data for {len(gps_lookup)} images")
        
        # Analysis configurations
        analysis_configs = [
            {
                "name": "Distorted Images Analysis",
                "cam1_folder": constants.CAM1_DISTORTED_FOLDER,
                "cam2_folder": constants.CAM2_DISTORTED_FOLDER,
                "cam1_type": "distorted",
                "cam2_type": "distorted",
                "cam1_model": "OPENCV_FISHEYE",
                "cam2_model": "OPENCV_FISHEYE",
                "query_type": "distorted"
            },
            {
                "name": "Undistorted Images Analysis",
                "cam1_folder": constants.CAM1_UNDISTORTED_FOLDER,
                "cam2_folder": constants.CAM2_UNDISTORTED_FOLDER,
                "cam1_type": "undistorted",
                "cam2_type": "undistorted",
                "cam1_model": "SIMPLE_RADIAL",
                "cam2_model": "SIMPLE_RADIAL",
                "query_type": "undistorted"
            }
        ]
        
        for config in analysis_configs:
            print(f"\n{'='*60}")
            print(f"{config['name']}")
            print(f"{'='*60}")
            
            results_output.append(f"\n{'='*60}")
            results_output.append(f"{config['name']}")
            results_output.append(f"{'='*60}")
            
            # Process cam1 and cam2 separately
            for cam_name, folder in [("cam1", config["cam1_folder"]), 
                                    ("cam2", config["cam2_folder"])]:
                
                print(f"\nAnalyzing {cam_name}...")
                results_output.append(f"\nAnalyzing {cam_name}...")
                
                # Get all images from the folder (both jpg and png)
                all_images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                if not all_images:
                    print(f"No images found in {folder}")
                    print(f"  Checked for: {folder}/*.jpg and {folder}/*.png")
                    results_output.append(f"No images found in {folder}")
                    continue
                
                # Sort images by their numeric part
                all_images.sort(key=lambda x: int(get_original_numbered_file_name(x.name).split('.')[0]))
                
                print(f"Found {len(all_images)} images in {folder}")
                results_output.append(f"Found {len(all_images)} images in {folder}")
                
                # Sample a subset for analysis
                sample_images = all_images[::sample_rate]
                print(f"Sampling {len(sample_images)} images (every {sample_rate}th image)")
                results_output.append(f"Sampling {len(sample_images)} images (every {sample_rate}th image)")
                
                # Debug: print first few sampled images
                if sample_images:
                    print(f"  First sampled image: {sample_images[0].name}")
                    if len(sample_images) > 1:
                        print(f"  Second sampled image: {sample_images[1].name}")
                    results_output.append(f"  First sampled image: {sample_images[0].name}")
                
                all_errors = []
                missing_gps_count = 0
                no_matches_count = 0
                processed_count = 0
                
                for i, query_image in enumerate(sample_images):
                    # Get query image location
                    query_name = query_image.name
                    
                    query_name = query_image.name
                    original_filename = get_original_numbered_file_name(query_name)

                    if original_filename not in gps_lookup:
                        missing_gps_count += 1
                        continue
                    
                    query_easting, query_northing = gps_lookup[original_filename]
                    
                    # Show progress
                    if (i + 1) % 10 == 0:
                        print(f"  Processing image {i + 1}/{len(sample_images)}...")
                    
                    # Retrieve matches
                    matches = retrieve_top_matches(
                        query_image_path=query_image,
                        cam1_folder=config["cam1_folder"],
                        cam2_folder=config["cam2_folder"],
                        cam1_camera_model=config["cam1_model"],
                        cam2_camera_model=config["cam2_model"],
                        cam1_type=config["cam1_type"],
                        cam2_type=config["cam2_type"],
                        query_image_type=config["query_type"],
                        top_n=5
                    )
                    
                    if not matches:
                        no_matches_count += 1
                        continue
                    
                    processed_count += 1
                    
                    # Calculate errors for each match
                    match_errors = []
                    for match_name, inliers in matches:
                        original_filename = get_original_numbered_file_name(match_name)
                        if original_filename in gps_lookup:
                            match_easting, match_northing = gps_lookup[original_filename]
                            
                            # Calculate Euclidean distance
                            error = np.sqrt((query_easting - match_easting)**2 + 
                                        (query_northing - match_northing)**2)
                            match_errors.append(error)
                    
                    if match_errors:
                        all_errors.extend(match_errors)
                
                # Calculate statistics
                if all_errors:
                    mean_error = np.mean(all_errors)
                    median_error = np.median(all_errors)
                    std_error = np.std(all_errors)
                    min_error = np.min(all_errors)
                    max_error = np.max(all_errors)
                    percentile_95 = np.percentile(all_errors, 95)
                    
                    # Print to console
                    print(f"\n{cam_name} Results:")
                    print(f"  Total images in folder: {len(all_images)}")
                    print(f"  Images sampled: {len(sample_images)}")
                    print(f"  Images with matches: {processed_count}")
                    print(f"  Images without matches: {no_matches_count}")
                    print(f"  Images missing GPS: {missing_gps_count}")
                    print(f"  Total error measurements: {len(all_errors)}")
                    print(f"\nError Statistics:")
                    print(f"  Mean error: {mean_error:.2f} meters")
                    print(f"  Median error: {median_error:.2f} meters")
                    print(f"  Std deviation: {std_error:.2f} meters")
                    print(f"  Min error: {min_error:.2f} meters")
                    print(f"  Max error: {max_error:.2f} meters")
                    print(f"  95th percentile: {percentile_95:.2f} meters")
                    
                    # Add to output file
                    results_output.append(f"\n{cam_name} Results:")
                    results_output.append(f"  Total images in folder: {len(all_images)}")
                    results_output.append(f"  Images sampled: {len(sample_images)}")
                    results_output.append(f"  Images with matches: {processed_count}")
                    results_output.append(f"  Images without matches: {no_matches_count}")
                    results_output.append(f"  Images missing GPS: {missing_gps_count}")
                    results_output.append(f"  Total error measurements: {len(all_errors)}")
                    results_output.append(f"\nError Statistics:")
                    results_output.append(f"  Mean error: {mean_error:.2f} meters")
                    results_output.append(f"  Median error: {median_error:.2f} meters")
                    results_output.append(f"  Std deviation: {std_error:.2f} meters")
                    results_output.append(f"  Min error: {min_error:.2f} meters")
                    results_output.append(f"  Max error: {max_error:.2f} meters")
                    results_output.append(f"  95th percentile: {percentile_95:.2f} meters")
                else:
                    print(f"\n{cam_name}: No valid matches found for error calculation")
                    print(f"  Total images checked: {len(sample_images)}")
                    print(f"  Images missing GPS: {missing_gps_count}")
                    print(f"  Images without matches: {no_matches_count}")
                    
                    results_output.append(f"\n{cam_name}: No valid matches found for error calculation")
                    results_output.append(f"  Total images checked: {len(sample_images)}")
                    results_output.append(f"  Images missing GPS: {missing_gps_count}")
                    results_output.append(f"  Images without matches: {no_matches_count}")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(results_output))
        
        print(f"\nResults saved to: {output_file}")
        print("\nRetrieval accuracy analysis completed")
        return True
        
    except Exception as e:
        print(f"Error during retrieval analysis: {e}")
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
        4: ("Build COLMAP Databases", lambda: build_colmap_databases(args.rebuild_colmap)),
        5: ("Analyze Retrieval Accuracy", lambda: analyze_retrieval_accuracy(args.sample_rate))
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