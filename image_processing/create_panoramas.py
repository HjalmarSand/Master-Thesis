import math
from os import path
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utility import get_original_numbered_file_name


def find_horizon_roll_angle(image: np.ndarray) -> float or None:
    """
    Analyzes an image to find the dominant horizon line and returns its roll angle.
    This is the core detection logic.
    """
    # --- Tuning Parameters (can be moved to a config file or constants at the top) ---
    CANNY_THRESHOLD_1 = 50
    CANNY_THRESHOLD_2 = 150
    BLUR_KERNEL_SIZE = (5, 5)
    HOUGH_THRESHOLD = 50
    HOUGH_MIN_LINE_LENGTH = 50
    HOUGH_MAX_LINE_GAP = 10
    ROI_BOTTOM_IGNORE_PERCENT = 0.35
    HORIZONTAL_ANGLE_TOLERANCE = 30.0

    (h, w) = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
    edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    mask = np.zeros_like(edges)
    roi_bottom_y = int(h * (1 - ROI_BOTTOM_IGNORE_PERCENT))
    cv2.rectangle(mask, (0, 0), (w, roi_bottom_y), 255, -1)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return None

    candidate_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < HORIZONTAL_ANGLE_TOLERANCE:
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            candidate_lines.append({"angle": angle, "length": length})

    if not candidate_lines:
        return None

    best_line_info = max(candidate_lines, key=lambda x: x["length"])
    return best_line_info["angle"]


def correct_image_roll(image: np.ndarray, roll_angle: float) -> np.ndarray:
    """Rotates an image to counteract a given roll angle."""
    if (
        roll_angle is None or abs(roll_angle) < 0.1
    ):  # Don't process if no angle or angle is tiny
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    correction_angle = -roll_angle

    M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    corrected_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )  # Black borders
    return corrected_image


def rotate_panorama_by_orientation(panorama_image, orientation_degrees):
    """
    Rotate a panoramic image horizontally based on orientation angle.

    For panoramic images, rotation is achieved by shifting pixels horizontally
    in a circular manner rather than using traditional rotation which would
    distort the panoramic view.

    Args:
        panorama_image (np.ndarray): Input panoramic image
        orientation_degrees (float): Orientation angle in degrees

    Returns:
        np.ndarray: Rotated panoramic image
    """
    if panorama_image is None:
        return None

    height, width = panorama_image.shape[:2]

    # Convert orientation to pixel shift
    # Assuming 360 degrees corresponds to the full width of the panorama
    pixel_shift = int((orientation_degrees / 360.0) * width)

    # Perform circular shift
    if pixel_shift != 0:
        # Use numpy roll for circular shift along width axis
        rotated_image = np.roll(panorama_image, pixel_shift, axis=1)
    else:
        rotated_image = panorama_image.copy()

    return rotated_image


def create_panorama(cam2_right_img, cam1_left_img, cam1_right_img, cam2_left_img):
    """
    Stitches four pre-loaded images into a single panoramic image.

    Args:
        cam2_right_img (np.ndarray): Image data for the Back-Right view.
        cam1_left_img (np.ndarray): Image data for the Front-Left view.
        cam1_right_img (np.ndarray): Image data for the Front-Right view.
        cam2_left_img (np.ndarray): Image data for the Back-Left view.

    Returns:
        np.ndarray: The stitched panoramic image.
    """
    images = [cam1_right_img, cam2_left_img, cam2_right_img, cam1_left_img]

    # Validate that all images have the same height for concatenation
    first_height = images[0].shape[0]
    if not all(img.shape[0] == first_height for img in images):
        print("Warning: Mismatched image heights. Cannot create final panorama.")
        return None

    # Concatenate the images horizontally
    panorama = np.concatenate(images, axis=1)
    return panorama


# TODO: This function is right now not updated to the new design of build_panoramas
# Before stitching front and back images, we can apply edge detection to correct the roll of the images
def create_panorama_edge_detection(
    cam1_left_path, cam1_right_path, cam2_left_path, cam2_right_path, verbose=False
):
    """
    Creates a roll-corrected 360-degree panorama from four 90-degree views.

    It first stitches front and back 180-degree views to detect the horizon roll,
    then corrects both individual images before the final stitching.

    Args:
        (paths to the four images)
        verbose (bool): If True, prints diagnostic messages for each step.

    Returns:
        np.ndarray: The stitched and roll-corrected panoramic image, or None if an error occurred.
    """

    print(
        "WARNING: THE FUNCTION create_panorama_edge_detection IS NOT UPDATED AND WILL BEHAVE UNEXPECTEDLY OR CRASH. "
    )
    print(
        "please set use_edge_detection=False in build_panoramas to use the old stitching method."
    )
    cam1_left = cv2.imread(str(cam1_left_path))
    cam1_right = cv2.imread(str(cam1_right_path))
    cam2_left = cv2.imread(str(cam2_left_path))
    cam2_right = cv2.imread(str(cam2_right_path))

    all_images = [cam1_left, cam1_right, cam2_left, cam2_right]
    if any(img is None for img in all_images):
        tqdm.write(
            "Warning: Could not load one or more images for stitching. Skipping."
        )
        return None

    # Assuming cam1 is front, cam2 is back
    front_view_180 = np.concatenate([cam1_left, cam1_right], axis=1)
    back_view_180 = np.concatenate(
        [cam2_right, cam2_left], axis=1
    )  # Note physical order

    front_roll_angle = find_horizon_roll_angle(front_view_180)
    back_roll_angle = find_horizon_roll_angle(back_view_180)

    if verbose:
        front_msg = (
            f"{front_roll_angle:.2f}°" if front_roll_angle is not None else "Not found"
        )
        back_msg = (
            f"{back_roll_angle:.2f}°" if back_roll_angle is not None else "Not found"
        )
        tqdm.write(f"  > Detected Rolls: Front={front_msg}, Back={back_msg}")

    # Correct each of the four original images
    corrected_c1_left = correct_image_roll(cam1_left, front_roll_angle)
    corrected_c1_right = correct_image_roll(cam1_right, front_roll_angle)
    corrected_c2_left = correct_image_roll(cam2_left, back_roll_angle)
    corrected_c2_right = correct_image_roll(cam2_right, back_roll_angle)

    # Stitch the final 360-degree panorama in the specified order
    final_images_in_order = [
        corrected_c2_right,
        corrected_c1_left,
        corrected_c1_right,
        corrected_c2_left,
    ]

    return np.concatenate(final_images_in_order, axis=1)


def build_panoramas(
    cam1_left_dir,
    cam1_right_dir,
    cam2_left_dir,
    cam2_right_dir,
    output_dir,
    orientation_csv_path=None,
    apply_rotation=True,
    manual_rotation_offset_deg=16.0,gi
):
    """
    Finds corresponding sets of four images, stitches them into intermediate
    and final panoramas, and saves them.

    - Creates and saves concatenated 'cam1' images to 'output_dir/undistorted_cam1/'.
    - Creates and saves concatenated 'cam2' images to 'output_dir/undistorted_cam2/'.
    - Creates and saves the final 360° panorama to 'output_dir/'.
    - Optionally applies orientation-based rotation to the final panorama.
    """
    # Define and create all necessary output directories
    output_path = Path(output_dir) / "panoramas"
    cam1_output_path = Path(output_dir) / "undistorted_cam1"
    cam2_output_path = Path(output_dir) / "undistorted_cam2"

    output_path.mkdir(parents=True, exist_ok=True)
    cam1_output_path.mkdir(parents=True, exist_ok=True)
    cam2_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Final panoramas will be saved to: {output_path}")
    print(f"Intermediate Cam1 images will be saved to: {cam1_output_path}")
    print(f"Intermediate Cam2 images will be saved to: {cam2_output_path}")

    # (The rest of the initial setup code is unchanged)
    # Load orientation data if provided
    orientation_lookup = {}
    if orientation_csv_path and apply_rotation:
        try:
            orientation_df = pd.read_csv(orientation_csv_path)
            if (
                "filename" in orientation_df.columns
                and "Orientation" in orientation_df.columns
            ):
                orientation_lookup = {
                    row["filename"]: row["Orientation"]
                    for _, row in orientation_df.iterrows()
                }
                print(f"Loaded orientation data from {orientation_csv_path}")
            else:
                print(
                    "Warning: CSV missing required columns (filename, Orientation). Skipping rotation."
                )
                apply_rotation = False
        except Exception as e:
            print(f"Warning: Could not load orientation CSV: {e}. Skipping rotation.")
            apply_rotation = False

    if manual_rotation_offset_deg != 0.0:
        print(
            f"Applying a constant manual rotation offset of {manual_rotation_offset_deg} to all panoramas."
        )

    # Use cam1_left_dir as the reference to find all numbers in the image names as identifier
    base_images = list(Path(cam2_left_dir).glob("*.jpg"))
    if not base_images:
        print(f"Error: No images found in the base directory: {cam1_left_dir}")
        return

    print(f"Found {len(base_images)} images to process for panoramas.")

    successful_panoramas = 0

    for base_image_path in tqdm(base_images, desc="Building Panoramas"):
        original_filename = get_original_numbered_file_name(base_image_path.name)
        if not original_filename:
            tqdm.write(
                f"Warning: Could not parse identifier from {base_image_path.name}. Skipping."
            )
            continue

        # Construct all file paths
        cam1_left_name = f"undistorted_cam1_{Path(original_filename).stem}_left.jpg"
        cam1_right_name = f"undistorted_cam1_{Path(original_filename).stem}_right.jpg"
        cam2_left_name = f"undistorted_cam2_{Path(original_filename).stem}_left.jpg"
        cam2_right_name = f"undistorted_cam2_{Path(original_filename).stem}_right.jpg"

        cam1_left_path = Path(cam1_left_dir) / cam1_left_name
        cam1_right_path = Path(cam1_right_dir) / cam1_right_name
        cam2_left_path = Path(cam2_left_dir) / cam2_left_name
        cam2_right_path = Path(cam2_right_dir) / cam2_right_name

        all_paths = [cam1_left_path, cam1_right_path, cam2_left_path, cam2_right_path]
        if not all(p.exists() for p in all_paths):
            tqdm.write(
                f"Warning: Missing one or more views for {original_filename}. Skipping."
            )
            continue

        img_c1_l = cv2.imread(str(cam1_left_path))
        img_c1_r = cv2.imread(str(cam1_right_path))
        img_c2_l = cv2.imread(str(cam2_left_path))
        img_c2_r = cv2.imread(str(cam2_right_path))

        all_images = [img_c1_l, img_c1_r, img_c2_l, img_c2_r]
        if any(img is None for img in all_images):
            tqdm.write(
                f"Warning: Failed to load one or more image files for {original_filename}. Skipping."
            )
            continue

        if img_c1_l.shape[0] == img_c1_r.shape[0]:  # Check height match
            cam1_pano = np.concatenate([img_c1_l, img_c1_r], axis=1)
            cam1_output_filename = (
                f"undistorted_cam1_{Path(original_filename).stem}.jpg"
            )
            cv2.imwrite(str(cam1_output_path / cam1_output_filename), cam1_pano)
        else:
            tqdm.write(
                f"Warning: Mismatched height for Cam1 images of {original_filename}. Skipping Cam1 pano."
            )

        if img_c2_l.shape[0] == img_c2_r.shape[0]:  # Check height match
            cam2_pano = np.concatenate([img_c2_l, img_c2_r], axis=1)
            cam2_output_filename = (
                f"undistorted_cam2_{Path(original_filename).stem}.jpg"
            )
            cv2.imwrite(str(cam2_output_path / cam2_output_filename), cam2_pano)
        else:
            tqdm.write(
                f"Warning: Mismatched height for Cam2 images of {original_filename}. Skipping Cam2 pano."
            )

        # --- CREATE FINAL 360 PANORAMA using pre-loaded images ---
        # The `use_edge_detection` flag is removed as the logic is now inside this function.
        # If you still need it, you would call `create_panorama_edge_detection` here instead.
        final_panorama = create_panorama(img_c2_r, img_c1_l, img_c1_r, img_c2_l)

        if final_panorama is not None:
            # Apply rotation logic (unchanged)
            if apply_rotation and original_filename in orientation_lookup:
                orientation = orientation_lookup[original_filename]
                total_rotation_angle = orientation + manual_rotation_offset_deg
                final_panorama = rotate_panorama_by_orientation(
                    final_panorama, total_rotation_angle
                )

            # Save the final panorama
            output_filename = f"panorama_{Path(original_filename).stem}.jpg"
            save_path = output_path / output_filename
            cv2.imwrite(str(save_path), final_panorama)
            successful_panoramas += 1
        else:
            tqdm.write(
                f"Warning: Could not create final panorama for {original_filename}."
            )

    return 0
