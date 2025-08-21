import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to Python's system path
sys.path.append(parent_dir)
import constants


def visualize_rotation_augmentation(
    image_path: Path, original_angle: float, rotation_to_apply: float
):
    """
    Loads an image, applies a specific rotation, and displays the
    original and transformed images side-by-side with their labels.
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load the original image
    original_image = Image.open(image_path)

    # 2. Apply the geometric augmentation (rotation)
    #    This is the exact same function call from your Dataset
    rotated_image = TF.rotate(original_image, angle=rotation_to_apply)

    # 3. Correct the label
    #    This is the crucial step you implemented correctly
    new_angle = original_angle + rotation_to_apply

    print("-" * 50)
    print(f"Image: {image_path.name}")
    print(f"Assumed Original Roll Angle: {original_angle:.2f}°")
    print(f"Applied Rotation: {rotation_to_apply:.2f}°")
    print(f"Corrected New Roll Angle: {new_angle:.2f}°")
    print("-" * 50)

    # 4. Visualize the result
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(original_image)
    axs[0].set_title(f"Original Image\nAssumed Roll: {original_angle:.2f}°")
    axs[0].axis("off")

    axs[1].imshow(rotated_image)
    axs[1].set_title(
        f"Augmented Image (+{rotation_to_apply:.2f}°)\nNew Correct Roll: {new_angle:.2f}°"
    )
    axs[1].axis("off")

    plt.suptitle("Testing Rotation Augmentation and Label Correction", fontsize=16)
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # --- PLEASE CHANGE THESE VALUES ---
    # 1. Point this to one of your training images
    IMAGE_TO_TEST = (
        constants.IMAGES_DIR / "dataset1/distorted_images_cam1/distorted_cam1_0801.jpg"
    )

    # 2. A pretend "ground truth" roll angle for this image
    ASSUMED_ORIGINAL_ROLL = 5.0

    # 3. The rotation angle we want to apply for this test
    ROTATION_ANGLE_TO_APPLY = -12.5  # Using a negative angle to test both directions
    # --- END OF VALUES TO CHANGE ---

    visualize_rotation_augmentation(
        image_path=IMAGE_TO_TEST,
        original_angle=ASSUMED_ORIGINAL_ROLL,
        rotation_to_apply=ROTATION_ANGLE_TO_APPLY,
    )
