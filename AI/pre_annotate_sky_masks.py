import json
import os
import sys

import cv2
import numpy as np
from PIL import Image  # For reading the original images to get dimensions

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to Python's system path
sys.path.append(parent_dir)

import constants

# TODO: Change when annotating other datasets
# Configuration
ORIGINAL_IMAGES_DIR = constants.IMAGES_DIR / "dataset3/panoramas"
BINARY_MASKS_DIR = constants.IMAGES_DIR / "dataset3/binary_panorama"
OUTPUT_DIR = constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations/dataset3"
SKY_LABEL_NAME = "sky"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

coco_data = {
    "info": {},
    "licenses": [],
    "categories": [{"id": 1, "name": SKY_LABEL_NAME, "supercategory": "background"}],
    "images": [],
    "annotations": [],
}
annotation_id = 0

# Get list of files, sorted for consistent pairing
original_image_files = sorted(
    [
        f
        for f in os.listdir(ORIGINAL_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)

for original_filename in original_image_files:
    base_name = os.path.splitext(original_filename)[0]  # e.g., "panorama_0801"

    mask_filename = f"binary_{base_name}.png"

    mask_path = os.path.join(BINARY_MASKS_DIR, mask_filename)

    if not os.path.exists(mask_path):
        print(
            f"Warning: No matching mask found at {mask_path} for {original_filename}. Skipping."
        )
        continue

    # Read the binary mask (as grayscale to ensure 1 channel)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: Could not read mask image {mask_path}. Skipping.")
        continue

    # Ensure mask is truly binary (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255  # Pixels > 0 become 255, others 0

    # Get original image dimensions for COCO metadata
    original_image_path = os.path.join(ORIGINAL_IMAGES_DIR, original_filename)
    with Image.open(original_image_path) as img:
        width, height = img.size

    image_id = len(coco_data["images"]) + 1  # Assign a unique ID for each image

    coco_data["images"].append(
        {
            "id": image_id,
            "file_name": original_filename,  # This MUST match the original image filename you upload to CVAT
            "width": width,
            "height": height,
        }
    )

    # Find contours from the binary mask
    # RETR_EXTERNAL retrieves only the outer contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) < 3:  # Need at least 3 points for a polygon
            continue

        segmentation = (
            contour.flatten().tolist()
        )  # Flatten points for COCO polygon format

        # Calculate bbox (x_min, y_min, width, height)
        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)
        if area == 0:  # Skip empty contours
            continue

        annotation_id += 1
        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # ID for 'sky' category (defined above as ID 1)
                "segmentation": [segmentation],  # COCO expects a list of polygons
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0,  # For semantic segmentation, typically 0
            }
        )

# Save to JSON
with open(os.path.join(OUTPUT_DIR, "annotations.json"), "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {os.path.join(OUTPUT_DIR, 'annotations.json')}")
print("\nRemember to:")
print("1. Create a CVAT task with a label named '{SKY_LABEL_NAME}' (type 'Mask').")
print("2. Upload your ORIGINAL images to the task.")
print(
    "3. After task creation, use 'Actions -> Upload annotations' and select 'COCO 1.0 (Segmentation)' with the generated 'annotations.json' file."
)
