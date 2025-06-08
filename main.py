import sys
import time
from pathlib import Path

from utils import split_stacked_images 
from utils.fisheye_undistorter import main as fisheye_undistorter_main
from utils import orientation_path_from_gps


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
        # Run the image splitting process
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
        # Run the fisheye undistortion process
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
        # Run the GPS orientation processing
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


def run_analysis():
    """
    Stage 4: Run data analysis (placeholder for future implementation).
    """
    print_stage_header(4, "DATA ANALYSIS", 
                    "Analyzing processed images and data")
    
    # TODO: Implement analysis
    print("Data analysis not implemented yet")
    return True  # Return True for now to continue pipeline


def main():
    """
    Main pipeline function that orchestrates the complete image processing workflow.
    """
    start_time = time.time()
    
    print_separator("FISHEYE IMAGE PROCESSING PIPELINE")
    print("Starting complete image processing pipeline...")
    print(f"Working directory: {Path.cwd()}")
    
    # Pipeline stages
    pipeline_stages = [
        ("Image Splitting", run_image_splitting),
        ("Fisheye Undistortion", run_fisheye_undistortion),
        ("GPS & Orientation Processing", run_gps_orientation_processing),
        ("Data Analysis", run_analysis),
    ]
    
    successful_stages = 0
    total_stages = len(pipeline_stages)
    
    # Execute pipeline stages
    for i, (stage_name, stage_function) in enumerate(pipeline_stages):
        try:
            success = stage_function()
            if success:
                successful_stages += 1
            else:
                print(f"{stage_name} failed")
                print("Pipeline stopped due to failure")
                break
                
        except KeyboardInterrupt:
            print(f"\nPipeline interrupted by user during {stage_name}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error in {stage_name}: {e}")
            print("Pipeline stopped due to error")
            break
    
    # Pipeline summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print_separator("PIPELINE SUMMARY")
    print(f"Completed stages: {successful_stages}/{total_stages}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    if successful_stages == total_stages:
        print("All pipeline stages completed successfully!")
        return 0
    else:
        print("Pipeline completed with some failures")
        return 1


def run_single_stage(stage_name):
    """
    Run a single pipeline stage by name (useful for debugging/development).
    """
    stage_map = {
        "split": run_image_splitting,
        "undistort": run_fisheye_undistortion,
        "gps": run_gps_orientation_processing,
        "analyze": run_analysis,
    }
    
    if stage_name.lower() not in stage_map:
        print(f"Unknown stage: {stage_name}")
        print(f"Available stages: {list(stage_map.keys())}")
        return 1
    
    print_separator(f"RUNNING SINGLE STAGE: {stage_name.upper()}")
    success = stage_map[stage_name.lower()]()
    
    if success:
        print(f"Stage '{stage_name}' completed successfully!")
        return 0
    else:
        print(f"Stage '{stage_name}' failed!")
        return 1


if __name__ == "__main__":
    # Check for command line arguments to run specific stages
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stage" and len(sys.argv) > 2:
            # Run a single stage: python main.py --stage split
            exit_code = run_single_stage(sys.argv[2])
        elif sys.argv[1] in ["--help", "-h"]:
            print("Fisheye Image Processing Pipeline")
            print("Usage:")
            print("  python main.py                    # Run full pipeline")
            print("  python main.py --stage <name>    # Run single stage")
            print("  python main.py --help            # Show this help")
            print("\nAvailable stages:")
            print("  split     - Split stacked images")
            print("  undistort - Undistort fisheye images")
            print("  gps       - Process GPS/orientation data")
            print("  analyze   - Run data analysis")
            exit_code = 0
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            exit_code = 1
    else:
        # Run the full pipeline
        exit_code = main()
    
    sys.exit(exit_code)