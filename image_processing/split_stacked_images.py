import cv2
from pathlib import Path
import constants
from .utility import get_image_dimensions_opencv

def get_split_dimensions(sample_image_path: Path):
    """Get dimensions for splitting images, with fallback to constants."""
    try:
        width, stacked_height = get_image_dimensions_opencv(str(sample_image_path))
        print(f"  -> Detected stacked image dimensions from '{sample_image_path.name}': "
            f"Width={width}, Stacked Height={stacked_height}")
        
        if hasattr(constants, 'SPLIT_IMG_WIDTH') and hasattr(constants, 'SPLIT_IMG_HEIGHT'):
            const_width = constants.SPLIT_IMG_WIDTH
            const_stacked_height = constants.SPLIT_IMG_HEIGHT * 2
            if width != const_width or stacked_height != const_stacked_height:
                print(f"     Warning: Detected dimensions ({width}x{stacked_height}) differ from constants "
                    f"({const_width}x{const_stacked_height}). Using detected dimensions.")
        
        return width, stacked_height
        
    except Exception as e:
        print(f"  -> Error getting dimensions from {sample_image_path}: {e}")
        fallback_width = constants.SPLIT_IMG_WIDTH
        fallback_height = constants.SPLIT_IMG_HEIGHT * 2
        print(f"     Falling back to constants: {fallback_width}x{fallback_height}")
        return fallback_width, fallback_height

def create_output_directories(dataset_path: Path):
    """
    Create output directories using constants.IMAGES_DIR as the root.
    The output is placed in a new folder named after the original dataset.
    """
    try:
        # Use the path from constants as the output root.
        output_root = Path(constants.IMAGES_DIR)

        # Create the new base directory named after the dataset
        output_base_dir = output_root / dataset_path.name
        
        # Define the final output directories
        cam1_dir = output_base_dir / "distorted_images_cam1"
        cam2_dir = output_base_dir / "distorted_images_cam2"
        
        cam1_dir.mkdir(parents=True, exist_ok=True)
        cam2_dir.mkdir(parents=True, exist_ok=True)
        
        return cam1_dir, cam2_dir
        
    except Exception as e:
        print(f"  -> ERROR: Could not create output directories: {e}")
        return None, None

def split_single_image(image_path: Path, width: int, height_cam1: int, stacked_height: int, cam1_dir: Path, cam2_dir: Path):
    """Split a single stacked image into two camera images."""
    try:
        composite_img = cv2.imread(str(image_path))
        if composite_img is None:
            print(f"    - ERROR: Could not read {image_path.name}")
            return False
        
        actual_height, actual_width = composite_img.shape[:2]
        if actual_width != width or actual_height != stacked_height:
            print(f"    - WARNING: {image_path.name} has unexpected dimensions "
                f"({actual_width}x{actual_height}), expected ({width}x{stacked_height}). Skipping.")
            return False

        img_cam1 = composite_img[0:height_cam1, 0:width]
        img_cam2 = composite_img[height_cam1:stacked_height, 0:width]
        
        base_name = image_path.stem
        output_cam1 = cam1_dir / f"distorted_cam1_{base_name}.jpg"
        output_cam2 = cam2_dir / f"distorted_cam2_{base_name}.jpg"
        
        cv2.imwrite(str(output_cam1), img_cam1)
        cv2.imwrite(str(output_cam2), img_cam2)
        
        return True
        
    except Exception as e:
        print(f"    - ERROR: Exception while processing {image_path.name}: {e}")
        return False

def process_dataset_folder(dataset_path: Path):
    """Finds, processes, and splits all images within a single dataset folder."""
    print(f"\nProcessing Dataset: {dataset_path.name}")
    images_dir = dataset_path / "images"

    if not images_dir.is_dir():
        print(f"  -> SKIPPING: 'images' subfolder not found in '{dataset_path.name}'.")
        return 0, 0

    images_to_process = sorted(list(images_dir.glob("*.jpg")))
    if not images_to_process:
        print(f"  -> SKIPPING: No .jpg files found in '{images_dir}'.")
        return 0, 0
    
    print(f"  -> Found {len(images_to_process)} .jpg images.")

    width, stacked_height = get_split_dimensions(images_to_process[0])
    height_cam1 = stacked_height // 2
    
    cam1_dir, cam2_dir = create_output_directories(dataset_path)
    if not cam1_dir or not cam2_dir:
        return 0, len(images_to_process)

    print(f"  -> Outputting to: '{cam1_dir.parent}'")

    successful_splits = 0
    for image_path in images_to_process:
        if split_single_image(image_path, width, height_cam1, stacked_height, cam1_dir, cam2_dir):
            successful_splits += 1
    
    failed_splits = len(images_to_process) - successful_splits
    print(f"  -> Finished: {successful_splits} successful, {failed_splits} failed.")
    return successful_splits, failed_splits

def run_splitter(root_folder_path: Path):
    """
    Main function to orchestrate the image splitting process for all
    dataset folders within a root directory.
    """
    print("="*40)
    print("      STARTING BATCH IMAGE SPLITTING      ")
    print("="*40)

    if not root_folder_path.is_dir():
        print(f"FATAL: Root directory not found at '{root_folder_path}'")
        return 1

    dataset_folders = [d for d in root_folder_path.iterdir() if d.is_dir()]
    
    if not dataset_folders:
        print(f"Error: No dataset folders found in '{root_folder_path}'")
        return 1

    total_success = 0
    total_fails = 0

    for dataset_path in dataset_folders:
        success, fails = process_dataset_folder(dataset_path)
        total_success += success
        total_fails += fails
    
    print("\n" + "="*40)
    print("          BATCH PROCESS COMPLETE          ")
    print(f"Summary across {len(dataset_folders)} dataset(s):")
    print(f"  Total images processed successfully: {total_success}")
    print(f"  Total images failed: {total_fails}")
    print("="*40)
    
    return 0 if total_fails == 0 else 1