import cv2
import os
import glob
import config
from utility import get_image_dimensions_opencv

images_in_dataset = glob.glob(os.path.join(config.DATASET1_DIR, "*.jpg"))
if not images_in_dataset:
    print(f"Error: No JPG image files found in {config.DATASET1_DIR} to determine dimensions.")
    exit()
image_path_for_dimensions = images_in_dataset[0] # Take the first one found

(width, stacked_height) = get_image_dimensions_opencv(image_path_for_dimensions)
print(f"Detected stacked image dimensions from '{os.path.basename(image_path_for_dimensions)}': Width={width}, Stacked Height={stacked_height}")

height_cam1 = stacked_height // 2

print(f"Calculated output dimensions for cam1 images: {width}x{height_cam1}")
print(f"Calculated output dimensions for cam2 images: {width}x{stacked_height - height_cam1}")

cam1_folder = os.path.join(config.COLMAP_DIR, "images_cam1")
cam2_folder = os.path.join(config.COLMAP_DIR, "images_cam2")

os.makedirs(cam1_folder, exist_ok=True)
os.makedirs(cam2_folder, exist_ok=True)

for path in images_in_dataset:
    base_filename = os.path.basename(path)
    print(f"Processing {base_filename}...")

    composite_img = cv2.imread(path)
    if composite_img is None:
        print(f"  Could not read {path}, skipping.")
        continue

    img_cam1 = composite_img[0:height_cam1, 0:width]

    # Crop for Camera 2 (bottom part)
    # Starts from where cam1 ended and goes to the full stacked_height
    img_cam2 = composite_img[height_cam1:stacked_height, 0:width]

    output_filename_cam1 = os.path.join(cam1_folder, f"cam1_{base_filename}")
    output_filename_cam2 = os.path.join(cam2_folder, f"cam2_{base_filename}")

    cv2.imwrite(output_filename_cam1, img_cam1)
    cv2.imwrite(output_filename_cam2, img_cam2)

print("--- Image splitting complete. ---")
print(f"Images for cam1 saved to: {cam1_folder}")
print(f"Images for cam2 saved to: {cam2_folder}")