import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .utility import get_original_numbered_file_name

# Use to test only one quadruplet of images
def create_panorama(cam1_left_path, cam1_right_path, cam2_left_path, cam2_right_path):
    """
    Loads four rectilinear views and stitches them into a single panoramic image.
    
    The order of stitching is based on the assumed camera rig geometry:
    - Back-Right (-135 deg)
    - Front-Left (-45 deg)
    - Front-Right (+45 deg)
    - Back-Left (+135 deg)
    
    Args:
        cam1_left_path (Path): Path to the front-left view.
        cam1_right_path (Path): Path to the front-right view.
        cam2_left_path (Path): Path to the back-left view.
        cam2_right_path (Path): Path to the back-right view.
        
    Returns:
        np.ndarray: The stitched panoramic image, or None if an error occurred.
    """

    image_paths_in_order = [
        cam2_right_path,  # Back-Right view
        cam1_left_path,   # Front-Left view
        cam1_right_path,  # Front-Right view
        cam2_left_path,   # Back-Left view
    ]

    images = [cv2.imread(str(p)) for p in image_paths_in_order]

    # Validate that all images were loaded successfully
    if any(img is None for img in images):
        print(f"Warning: Could not load one or more images for stitching. Skipping.")
        return None

    # Validate that all images have the same height for concatenation
    first_height = images[0].shape[0]
    if not all(img.shape[0] == first_height for img in images):
        print("Warning: Mismatched image heights. Cannot stitch.")
        return None

    # Concatenate the images horizontally
    panorama = np.concatenate(images, axis=1)
    
    return panorama

# Use to generate batch panoramas
def build_panoramas(cam1_left_dir, cam1_right_dir, cam2_left_dir, cam2_right_dir, output_dir):
    """
    Finds all corresponding sets of four images and stitches them into panoramas.
    
    Args:
        cam1_left_dir (Path): Directory for front-left views.
        cam1_right_dir (Path): Directory for front-right views.
        cam2_left_dir (Path): Directory for back-left views.
        cam2_right_dir (Path): Directory for back-right views.
        output_dir (Path): Directory to save the final panoramas.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use cam1_left_dir as the reference to find all numbers in the image names as identifier
    base_images = list(Path(cam1_left_dir).glob("*.jpg"))
    if not base_images:
        print(f"Error: No images found in the base directory: {cam1_left_dir}")
        return
        
    print(f"Found {len(base_images)} images to process for panoramas.")

    for base_image_path in tqdm(base_images, desc="Building Panoramas"):
        # Extract the original file number (e.g., "0894.jpg")
        original_filename = get_original_numbered_file_name(base_image_path.name)
        if not original_filename:
            tqdm.write(f"Warning: Could not parse numerical identifier from {base_image_path.name}. Skipping.")
            continue
        
        # Construct the paths for the other three images
        cam1_left_name = f"undistorted_cam1_{Path(original_filename).stem}_left.jpg"
        cam1_right_name = f"undistorted_cam1_{Path(original_filename).stem}_right.jpg"
        cam2_left_name = f"undistorted_cam2_{Path(original_filename).stem}_left.jpg"
        cam2_right_name = f"undistorted_cam2_{Path(original_filename).stem}_right.jpg"

        cam1_left_path = Path(cam1_left_dir) / cam1_left_name
        cam1_right_path = Path(cam1_right_dir) / cam1_right_name
        cam2_left_path = Path(cam2_left_dir) / cam2_left_name
        cam2_right_path = Path(cam2_right_dir) / cam2_right_name

        # Check if all four files exist before trying to stitch
        all_paths = [cam1_left_path, cam1_right_path, cam2_left_path, cam2_right_path]
        if not all(p.exists() for p in all_paths):
            tqdm.write(f"Warning: Missing one or more views for {original_filename}. Skipping.")
            continue

        # Create the panorama for this timestamp
        panorama = create_panorama(*all_paths)
        
        if panorama is not None:
            # Save the result
            output_filename = f"panorama_{Path(original_filename).stem}.jpg"
            save_path = output_path / output_filename
            cv2.imwrite(str(save_path), panorama)
            
    print(f"Panorama creation complete. Files saved to: {output_path}")
    return 0