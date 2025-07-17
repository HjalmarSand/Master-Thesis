import math

import cv2
import numpy as np

# --- TUNING PARAMETERS ---
# Canny Edge Detector thresholds
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Gaussian Blur kernel size (must be an odd number)
BLUR_KERNEL_SIZE = (5, 5)

# Hough Line Transform parameters
HOUGH_THRESHOLD = 50  # Minimum number of votes to consider a line
HOUGH_MIN_LINE_LENGTH = 50  # Minimum length of a line
HOUGH_MAX_LINE_GAP = 10  # Maximum allowed gap between points on the same line

# --- Region of Interest (ROI) Parameters ---
# We create a trapezoid, ignoring the top and bottom portions of the image.
# This helps to exclude clouds at the top and boat wakes/foreground at the bottom.
# Values are percentages of the total image height.
ROI_TOP_IGNORE_PERCENT = 0.33  # Ignore top 33% of the image (sky/clouds)
ROI_BOTTOM_IGNORE_PERCENT = 0.4  # Ignore bottom 25% of the image (boat/wake)

# --- Horizon Line Filtering ---
# Angle range (in degrees) to consider a line as a potential horizon candidate.
# A narrow range like [-15, 15] is good for rejecting non-horizon lines.
HORIZON_ANGLE_RANGE = (-15.0, 15.0)


def calculate_weighted_horizon_angle(lines):
    """
    Calculates a single, robust roll angle from a list of detected lines.
    It does this by computing a weighted average of the angles, where the
    weight of each line is its length.

    Args:
        lines: A list of lines detected by HoughLinesP.

    Returns:
        The final calculated roll angle in degrees, or 0.0 if no valid lines are found.
    """
    if lines is None:
        return 0.0

    candidate_lines = []
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        # Calculate angle and length
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Filter lines to be within the specified angle range (near horizontal)
        if HORIZON_ANGLE_RANGE[0] < angle < HORIZON_ANGLE_RANGE[1]:
            candidate_lines.append({"angle": angle, "length": length})

    if not candidate_lines:
        print("Warning: No candidate horizon lines found.")
        return 0.0

    # Calculate the weighted average of the angles
    weighted_angle_sum = sum(line["angle"] * line["length"] for line in candidate_lines)
    total_length = sum(line["length"] for line in candidate_lines)

    return weighted_angle_sum / total_length


def rotate_and_resize(image, angle_degrees):
    """
    Rotates an image by a given angle and resizes the canvas to fit the
    entire rotated image without cropping the corners.

    Args:
        image: The original image to rotate.
        angle_degrees: The angle to rotate the image by, in degrees.

    Returns:
        The rotated image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Calculate the new bounding box size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    return cv2.warpAffine(
        image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def process_and_correct_image(image_path1, image_path2):
    """
    Main pipeline: Loads an image, detects the horizon, and corrects the roll.
    """
    original_image1 = cv2.imread(image_path1)
    original_image2 = cv2.imread(image_path2)

    original_image = cv2.hconcat([original_image2, original_image1])
    if original_image is None:
        print(f"Error: Could not load image at {image_path1} or {image_path2}")
        return

    # --- 1. PRE-PROCESSING ---
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)

    # --- 2. CANNY EDGE DETECTION ---
    edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # --- 3. CREATE AND APPLY REGION OF INTEREST (ROI) MASK ---
    h, w = edges.shape
    mask = np.zeros_like(edges)

    # Define the vertices of the trapezoidal ROI
    roi_top_y = int(h * ROI_TOP_IGNORE_PERCENT)
    roi_bottom_y = int(h * (1 - ROI_BOTTOM_IGNORE_PERCENT))
    roi_corners = np.array(
        [[(0, roi_top_y), (w, roi_top_y), (w, roi_bottom_y), (0, roi_bottom_y)]],
        dtype=np.int32,
    )

    # Fill the ROI polygon with white
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # --- 4. HOUGH LINE TRANSFORM on the masked edges ---
    lines = cv2.HoughLinesP(
        masked_edges,
        1,  # rho resolution (1 pixel)
        np.pi / 180,  # theta resolution (1 degree)
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    # --- 5. CALCULATE THE FINAL HORIZON ANGLE ---
    roll_angle = calculate_weighted_horizon_angle(lines)
    print(f"Horizon detected! Estimated Roll Angle: {roll_angle:.2f} degrees")

    # --- 6. CORRECT THE IMAGE ---
    # The correction angle is the negative of the detected roll angle
    corrected_image = rotate_and_resize(original_image, -roll_angle)

    # --- 7. VISUALIZATION (for debugging) ---
    debug_image = original_image.copy()
    # Draw the ROI
    cv2.polylines(
        debug_image, [roi_corners], isClosed=True, color=(0, 0, 255), thickness=2
    )
    # Draw all detected lines (within the ROI) in blue
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw the final calculated horizon line in green
    center_y = h // 2
    length = w
    angle_rad = math.radians(roll_angle)
    x1 = int((w / 2) - (length / 2) * math.cos(angle_rad))
    y1 = int(center_y - (length / 2) * math.sin(angle_rad))
    x2 = int((w / 2) + (length / 2) * math.cos(angle_rad))
    y2 = int(center_y + (length / 2) * math.sin(angle_rad))
    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    cv2.putText(
        debug_image,
        f"Angle: {roll_angle:.2f} deg",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Masked Edges within ROI", masked_edges)
    cv2.imshow("Horizon Detection (Debug)", debug_image)
    cv2.imshow("Corrected Image", corrected_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Replace this with the path to your image
    # Use an image of an ocean with islands on the horizon for best results.
    TEST_IMAGE_PATH1 = (
        "images/dataset1/undistorted_cam1_right/undistorted_cam1_0801_right.jpg"
    )
    TEST_IMAGE_PATH2 = (
        "images/dataset1/undistorted_cam1_left/undistorted_cam1_0801_left.jpg"
    )

    process_and_correct_image(TEST_IMAGE_PATH1, image_path2=TEST_IMAGE_PATH2)
