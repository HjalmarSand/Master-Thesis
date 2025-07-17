import math
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
        cam1_left_path,  # Front-Left view
        cam1_right_path,  # Front-Right view
        cam2_left_path,  # Back-Left view
    ]

    images = [cv2.imread(str(p)) for p in image_paths_in_order]

    # Validate that all images were loaded successfully
    if any(img is None for img in images):
        print("Warning: Could not load one or more images for stitching. Skipping.")
        return None

    # Validate that all images have the same height for concatenation
    first_height = images[0].shape[0]
    if not all(img.shape[0] == first_height for img in images):
        print("Warning: Mismatched image heights. Cannot stitch.")
        return None

    # Concatenate the images horizontally
    panorama = np.concatenate(images, axis=1)

    return panorama


# Use to test only one quadruplet of images
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
    manual_rotation_offset_deg=31.0,  # + 5 more for dataset 2 and 3
    use_edge_detection=False,
):  # With our undistortion there seems to be about a 16 degree offset between the boats and the cameras orientation
    """
    Finds all corresponding sets of four images and stitches them into panoramas.
    Optionally applies orientation-based rotation and a manual correction offset.

    Args:
        cam1_left_dir (Path): Directory for front-left views.
        cam1_right_dir (Path): Directory for front-right views.
        cam2_left_dir (Path): Directory for back-left views.
        cam2_right_dir (Path): Directory for back-right views.
        output_dir (Path): Directory to save the final panoramas.
        orientation_csv_path (Path, optional): Path to CSV with orientation data.
        apply_rotation (bool): Whether to apply rotation if CSV is provided.
        manual_rotation_offset_deg (float): Manual angle in degrees to add to every rotation.
                                            Positive values rotate right, negative values rotate left.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load orientation data if provided
    orientation_lookup = {}
    if orientation_csv_path and apply_rotation:
        try:
            orientation_df = pd.read_csv(orientation_csv_path)
            print(f"Loaded orientation data for {len(orientation_df)} entries")

            if (
                "filename" in orientation_df.columns
                and "Orientation" in orientation_df.columns
            ):
                orientation_lookup = {
                    row["filename"]: row["Orientation"]
                    for _, row in orientation_df.iterrows()
                }
                print(
                    f"Will apply rotation based on orientation data from {orientation_csv_path}"
                )
            else:
                print(
                    "Warning: CSV missing required columns (filename, Orientation). Skipping rotation."
                )
                apply_rotation = False
        except Exception as e:
            print(f"Warning: Could not load orientation CSV: {e}. Skipping rotation.")
            apply_rotation = False
    elif orientation_csv_path is None and apply_rotation:
        print("No orientation CSV provided. Skipping rotation.")
        apply_rotation = False

    if manual_rotation_offset_deg != 0.0:
        print(
            f"Applying a constant manual rotation offset of {manual_rotation_offset_deg} to all panoramas."
        )

    # Use cam1_left_dir as the reference to find all numbers in the image names as identifier
    base_images = list(Path(cam2_left_dir).glob("*.jpg"))
    if not base_images:
        print(f"Error: No images found in the base directory: {cam2_left_dir}")
        return

    print(f"Found {len(base_images)} images to process for panoramas.")

    successful_panoramas = 0
    successful_rotations = 0
    missing_orientation = 0

    for base_image_path in tqdm(base_images, desc="Building Panoramas"):
        original_filename = get_original_numbered_file_name(base_image_path.name)
        if not original_filename:
            tqdm.write(
                f"Warning: Could not parse numerical identifier from {base_image_path.name}. Skipping."
            )
            continue

        cam1_left_name = f"undistorted_cam1_{Path(original_filename).stem}_left.jpg"
        cam1_right_name = f"undistorted_cam1_{Path(original_filename).stem}_right.jpg"
        cam2_left_name = f"undistorted_cam2_{Path(original_filename).stem}_left.jpg"
        cam2_right_name = f"undistorted_cam2_{Path(original_filename).stem}_right.jpg"

        cam1_left_path = Path(cam1_left_dir) / cam1_left_name
        cam1_right_path = Path(cam1_right_dir) / cam1_right_name
        cam2_left_path = Path(cam2_left_dir) / cam2_left_name
        cam2_right_path = Path(cam2_right_dir) / cam2_right_name

        all_paths = [cam2_left_path, cam2_right_path, cam1_left_path, cam1_right_path]
        if not all(p.exists() for p in all_paths):
            tqdm.write(
                f"Warning: Missing one or more views for {original_filename}. Skipping."
            )
            continue

        if use_edge_detection:
            panorama = create_panorama_edge_detection(*all_paths)
        else:
            panorama = create_panorama(*all_paths)

        if panorama is not None:
            successful_panoramas += 1

            if apply_rotation and original_filename in orientation_lookup:
                orientation = orientation_lookup[original_filename]

                total_rotation_angle = orientation + manual_rotation_offset_deg

                panorama = rotate_panorama_by_orientation(
                    panorama, total_rotation_angle
                )
                successful_rotations += 1
            elif apply_rotation and original_filename not in orientation_lookup:
                missing_orientation += 1
                tqdm.write(
                    f"Warning: No orientation data found for {original_filename}"
                )

            output_filename = f"panorama_{Path(original_filename).stem}.jpg"
            save_path = output_path / output_filename
            cv2.imwrite(str(save_path), panorama)

    print(f"Panorama creation complete. Files saved to: {output_path}")
    print(f"Successfully created: {successful_panoramas} panoramas")

    if apply_rotation:
        print(f"Successfully rotated: {successful_rotations} panoramas")
        if missing_orientation > 0:
            print(f"Missing orientation data for: {missing_orientation} panoramas")

    return 0
