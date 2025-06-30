import cv2
import os
import glob
import sys
from pathlib import Path
import constants
from utils.utility import get_image_dimensions_opencv


def validate_directories():
    """Validate that the required directories exist and are accessible."""
    if not constants.DATASET1_DIR.exists():
        print(f"Error: Dataset directory does not exist: {constants.DATASET1_DIR}")
        return False
    
    if not constants.DATASET1_DIR.is_dir():
        print(f"Error: Dataset path is not a directory: {constants.DATASET1_DIR}")
        return False
    
    return True


def find_stacked_images():
    """Find and validate .jpg images in the dataset directory."""
    # Only look for .jpg files as specified
    images_found = list(constants.DATASET1_DIR.glob("*.jpg"))
    
    if not images_found:
        print(f"Error: No .jpg image files found in {constants.DATASET1_DIR}")
        return []
    
    print(f"Found {len(images_found)} .jpg images to process")
    return images_found


def get_split_dimensions(sample_image_path):
    """Get dimensions for splitting images, with fallback to constants."""
    try:
        width, stacked_height = get_image_dimensions_opencv(str(sample_image_path))
        print(f"Detected stacked image dimensions from '{sample_image_path.name}': "
              f"Width={width}, Stacked Height={stacked_height}")
        
        # Update constants if they don't match detected dimensions
        if width != constants.SPLIT_IMG_WIDTH or stacked_height != (constants.SPLIT_IMG_HEIGHT * 2):
            print(f"Warning: Detected dimensions ({width}x{stacked_height}) differ from constants "
                  f"({constants.SPLIT_IMG_WIDTH}x{constants.SPLIT_IMG_HEIGHT * 2})")
            print("Using detected dimensions for processing")
        
        return width, stacked_height
        
    except Exception as e:
        print(f"Error getting dimensions from {sample_image_path}: {e}")
        print(f"Falling back to constants: {constants.SPLIT_IMG_WIDTH}x{constants.SPLIT_IMG_HEIGHT * 2}")
        return constants.SPLIT_IMG_WIDTH, constants.SPLIT_IMG_HEIGHT * 2


def create_output_directories():
    """Create output directories for split images."""
    # Convert string paths to Path objects for consistency
    cam1_dir = Path(constants.CAM1_DISTORTED_FOLDER)
    cam2_dir = Path(constants.CAM2_DISTORTED_FOLDER)
    
    try:
        cam1_dir.mkdir(parents=True, exist_ok=True)
        cam2_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directories ready:")
        print(f"  CAM1: {cam1_dir}")
        print(f"  CAM2: {cam2_dir}")
        return cam1_dir, cam2_dir
        
    except Exception as e:
        print(f"Error creating output directories: {e}")
        return None, None


def split_single_image(image_path, width, height_cam1, stacked_height, cam1_dir, cam2_dir):
    """Split a single stacked image into two camera images."""
    try:
        # Read the composite image
        composite_img = cv2.imread(str(image_path))
        if composite_img is None:
            print(f"  ERROR: Could not read {image_path.name}")
            return False
        
        # Validate image dimensions match expected
        actual_height, actual_width = composite_img.shape[:2]
        if actual_width != width or actual_height != stacked_height:
            print(f"  WARNING: {image_path.name} has unexpected dimensions "
                f"({actual_width}x{actual_height}) instead of ({width}x{stacked_height})")
        
        # Split the image
        img_cam1 = composite_img[0:height_cam1, 0:width]
        img_cam2 = composite_img[height_cam1:stacked_height, 0:width]
        
        # Generate output filenames (keeping .jpg extension)
        base_name = image_path.stem  # filename without extension
        
        output_cam1 = cam1_dir / f"distorted_cam1_{base_name}.jpg"
        output_cam2 = cam2_dir / f"distorted_cam2_{base_name}.jpg"
        
        # Save the split images
        success1 = cv2.imwrite(str(output_cam1), img_cam1)
        success2 = cv2.imwrite(str(output_cam2), img_cam2)
        
        if not success1 or not success2:
            print(f"  ERROR: Failed to save split images for {image_path.name}")
            return False
        
        print(f"  SUCCESS: Split {image_path.name}")
        return True
        
    except Exception as e:
        print(f"  ERROR: Exception while processing {image_path.name}: {e}")
        return False


def main():
    """Main function to orchestrate the image splitting process."""
    print("=== Starting Image Splitting Process ===")
    
    # Validate setup
    if not validate_directories():
        sys.exit(1)
    
    # Find images to process
    images_to_process = find_stacked_images()
    if not images_to_process:
        sys.exit(1)
    
    # Get dimensions from the first image
    width, stacked_height = get_split_dimensions(images_to_process[0])
    height_cam1 = stacked_height // 2
    height_cam2 = stacked_height - height_cam1
    
    print(f"Split configuration:")
    print(f"  Total width: {width}")
    print(f"  CAM1 height: {height_cam1}")
    print(f"  CAM2 height: {height_cam2}")
    print(f"  Total stacked height: {stacked_height}")
    
    # Create output directories
    cam1_dir, cam2_dir = create_output_directories()
    if cam1_dir is None or cam2_dir is None:
        sys.exit(1)
    
    # Process all images
    successful_splits = 0
    failed_splits = 0
    
    print(f"\nProcessing {len(images_to_process)} images...")
    for image_path in images_to_process:
        if split_single_image(image_path, width, height_cam1, stacked_height, cam1_dir, cam2_dir):
            successful_splits += 1
        else:
            failed_splits += 1
    
    # Summary
    print(f"\n=== Image Splitting Complete ===")
    print(f"Successfully processed: {successful_splits} images")
    if failed_splits > 0:
        print(f"Failed to process: {failed_splits} images")
    print(f"CAM1 images saved to: {cam1_dir}")
    print(f"CAM2 images saved to: {cam2_dir}")
    
    return 0 if failed_splits == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)