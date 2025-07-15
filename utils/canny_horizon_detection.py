import math

import cv2
import numpy as np

# --- TUNING PARAMETERS ---
# Canny Edge Detector thresholds
CANNY_THRESHOLD_1 = 25
CANNY_THRESHOLD_2 = 65

# Gaussian Blur kernel size (must be an odd number)
BLUR_KERNEL_SIZE = (5, 5)

# Hough Line Transform parameters
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 5

# --- NEW: Region of Interest (ROI) Parameter ---
# Percentage of the image height to ignore from the bottom.
# A value of 0.3 means the bottom 30% of the image will be masked out.
# TUNE THIS VALUE based on where the wake typically appears in your images.
ROI_BOTTOM_IGNORE_PERCENT = 0.35  # Ignore bottom 35%


def process_single_image_with_roi(image_path):
    """
    Loads a single image and finds the horizon, ignoring the bottom part of the
    image to avoid detecting the boat's wake.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Get image dimensions
    (h, w) = original_image.shape[:2]

    # 1. Pre-processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)

    # 2. Canny Edge Detection (on the full blurred image)
    edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # --- 3. NEW: Create and Apply the ROI Mask ---
    # Create a blank mask with the same dimensions as the edge map
    mask = np.zeros_like(edges)

    # Define the ROI polygon. We want a rectangle covering the top part of the image.
    # The bottom boundary of the ROI is the key parameter.
    roi_top_y = 0  # Start from the top
    roi_bottom_y = int(h * (1 - ROI_BOTTOM_IGNORE_PERCENT))

    # We define a rectangle covering the top part of the image we want to keep
    roi_corners = np.array(
        [[(0, roi_top_y), (w, roi_top_y), (w, roi_bottom_y), (0, roi_bottom_y)]],
        dtype=np.int32,
    )

    # Fill the ROI polygon with white (255)
    cv2.fillPoly(mask, roi_corners, 255)

    # Apply the mask to the edge map using a bitwise AND operation.
    # This keeps only the edge pixels that are within the white area of the mask.
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. Hough Line Transform (run on the MASKED edge map)
    lines = cv2.HoughLinesP(
        masked_edges,  # Use the masked edges, not the original ones!
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    debug_image = original_image.copy()

    # Draw the ROI boundary on the debug image for visualization
    cv2.rectangle(
        debug_image, (0, roi_top_y), (w, roi_bottom_y), (0, 0, 255), 2
    )  # Red rectangle
    cv2.putText(
        debug_image,
        "Search Area (ROI)",
        (10, roi_bottom_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    # 5. Filter for the Horizon and Calculate Angle (same logic as before)
    roll_angle = 0.0  # Default to 0 if no line is found
    if lines is not None:
        candidate_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -30.0 < angle < 30.0:
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                candidate_lines.append(
                    {"line": line[0], "angle": angle, "length": length}
                )
                cv2.line(
                    debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2
                )  # Blue candidates

        if candidate_lines:
            best_line_info = max(candidate_lines, key=lambda x: x["length"])
            roll_angle = best_line_info["angle"]
            bx1, by1, bx2, by2 = best_line_info["line"]
            cv2.line(
                debug_image, (bx1, by1), (bx2, by2), (0, 255, 0), 3
            )  # Green best line
            print(f"Horizon detected! Estimated Roll Angle: {roll_angle:.2f} degrees")
    else:
        print("Warning: No candidate lines found in the ROI.")

    # 6. Correct the image
    center = (w // 2, h // 2)
    correction_angle = -roll_angle
    M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    corrected_image = cv2.warpAffine(
        original_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )

    # Display the results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Masked Edges (Wake Removed)", masked_edges)
    cv2.imshow("Horizon Detection (Debug View)", debug_image)
    cv2.imshow("Corrected Image", corrected_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main Execution ---
if __name__ == "__main__":
    # Choose an image where the wake is causing problems
    TEST_IMAGE_PATH = "images/dataset3/distorted_images_cam1/distorted_cam1_2508.jpg"  # Replace with a real path

    process_single_image_with_roi(TEST_IMAGE_PATH)
