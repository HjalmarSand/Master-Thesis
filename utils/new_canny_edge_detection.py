import cv2
import matplotlib.pyplot as plt
import numpy as np


def correct_horizon(image_path: str, show_steps: bool = False):
    """
    Finds the horizon in an image with islands and corrects the camera roll.

    Args:
        image_path (str): Path to the input image.
        show_steps (bool): If True, displays intermediate steps for visualization.

    Returns:
        numpy.ndarray: The rotation-corrected image.
    """
    # 1. Load Image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Image not found.")
        return None

    # Make a copy for drawing lines on later
    image_with_line = original_image.copy()
    h, w = original_image.shape[:2]

    # 2. Preprocessing
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Edge Detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    if show_steps:
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(0)

    # 4. Region of Interest (ROI)
    # Focus on the middle vertical third of the image
    roi_top = h // 3
    roi_bottom = h * 2 // 3
    roi_mask = np.zeros_like(edges)
    roi_mask[roi_top:roi_bottom, :] = 255
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    if show_steps:
        cv2.imshow("ROI Edges", roi_edges)
        cv2.waitKey(0)

    # 5. Robust Line Fitting
    # Get the coordinates of all edge points in the ROI
    points = np.argwhere(roi_edges > 0)
    points = points[:, [1, 0]]  # Switch (row, col) to (x, y)

    if (
        len(points) < 20
    ):  # Heuristic threshold: not enough points to find a reliable line
        print(
            "Warning: Not enough edge points found to detect a horizon. Returning original image."
        )
        return original_image

    # Use cv2.fitLine to find the best-fit line.
    # It's robust to outliers (like the island bases).
    # It returns a direction vector (vx, vy) and a point on the line (x0, y0).
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    # 6. Angle Calculation
    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg[0]

    # The angle from fitLine can be in any quadrant. We want the angle
    # relative to the horizontal axis. A near-horizontal line will be
    # close to 0 or 180 degrees.
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    print(f"Detected horizon angle: {angle_deg:.2f} degrees")

    # Draw the detected line for visualization
    # We need to calculate two points on the line to draw it
    lefty = int((-x0 * vy / vx) + y0)
    righty = int(((w - x0) * vy / vx) + y0)
    cv2.line(image_with_line, (0, lefty), (w - 1, righty), (0, 0, 255), 2, cv2.LINE_AA)

    if show_steps:
        cv2.imshow("Detected Horizon Line", image_with_line)
        cv2.waitKey(0)

    # 7. Image Correction
    # Get the rotation matrix and perform the affine transformation
    center = (w // 2, h // 2)
    # We rotate by the negative of the detected angle to level the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Perform the rotation, filling in the new border pixels with black
    corrected_image = cv2.warpAffine(
        original_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return corrected_image


# --- --- --- ---
# How to use it
# --- --- --- ---
# Replace 'path/to/your/tilted_ocean.jpg' with your image file
image_path = "images/dataset1/undistorted_cam1_right/undistorted_cam1_0901_right.jpg"
corrected = correct_horizon(image_path, show_steps=True)

if corrected is not None:
    cv2.imshow("Corrected Image", corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
