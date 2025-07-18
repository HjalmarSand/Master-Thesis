import math

import cv2
import numpy as np
import pandas as pd

# --- TUNING PARAMETERS ---
# Canny Edge Detector thresholds
CANNY_THRESHOLD_1 = 60
CANNY_THRESHOLD_2 = 180

BLUR_KERNEL_SIZE = (5, 5)

HOUGH_THRESHOLD = 50  # Minimum number of votes to consider a line
HOUGH_MIN_LINE_LENGTH = 50  # Minimum length of a line
HOUGH_MAX_LINE_GAP = 10  # Maximum allowed gap between points on the same line

# Region of Interest Parameters
ROI_TOP_IGNORE_PERCENT = 0.40
ROI_BOTTOM_IGNORE_PERCENT = 0.45
ROI_LEFT_IGNORE_PERCENT = 0.4
ROI_RIGHT_IGNORE_PERCENT = 0.4

# Horizon Line Filtering
HORIZON_ANGLE_RANGE = (-15.0, 15.0)


def calculate_weighted_horizon_angle(lines):
    if lines is None:
        return 0.0
    candidate_lines = []
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if HORIZON_ANGLE_RANGE[0] < angle < HORIZON_ANGLE_RANGE[1]:
            candidate_lines.append({"angle": angle, "length": length})
    if not candidate_lines:
        return 0.0
    weighted_angle_sum = sum(line["angle"] * line["length"] for line in candidate_lines)
    total_length = sum(line["length"] for line in candidate_lines)
    return weighted_angle_sum / total_length if total_length > 0 else 0.0


def get_roll_angle_for_image(image_path):
    """
    Streamlined function for BATCH PROCESSING.
    Takes an image path, runs detection, and returns the calculated roll angle.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Warning: Could not load image at {image_path}. Skipping.")
        return None
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
    edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi_top_y = int(h * ROI_TOP_IGNORE_PERCENT)
    roi_bottom_y = int(h * (1 - ROI_BOTTOM_IGNORE_PERCENT))
    roi_left_x = int(w * ROI_LEFT_IGNORE_PERCENT)
    roi_right_x = int(w * (1 - ROI_RIGHT_IGNORE_PERCENT))
    roi_corners = np.array(
        [
            [
                (roi_left_x, roi_top_y),
                (roi_right_x, roi_top_y),
                (roi_right_x, roi_bottom_y),
                (roi_left_x, roi_bottom_y),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    return calculate_weighted_horizon_angle(lines)


def rotate_and_resize(image, angle_degrees):
    """Rotates an image and resizes the canvas to fit."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def visualize_single_image(image_path):
    """
    Function for VISUAL INVESTIGATION.
    Processes a single image and displays debug windows.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # --- Core processing logic (repeated here for clarity in this function) ---
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_KERNEL_SIZE, 0)
    edges = cv2.Canny(blurred_image, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi_top_y = int(h * ROI_TOP_IGNORE_PERCENT)
    roi_bottom_y = int(h * (1 - ROI_BOTTOM_IGNORE_PERCENT))
    roi_left_x = int(w * ROI_LEFT_IGNORE_PERCENT)
    roi_right_x = int(w * (1 - ROI_RIGHT_IGNORE_PERCENT))
    roi_corners = np.array(
        [
            [
                (roi_left_x, roi_top_y),
                (roi_right_x, roi_top_y),
                (roi_right_x, roi_bottom_y),
                (roi_left_x, roi_bottom_y),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    roll_angle = calculate_weighted_horizon_angle(lines)
    print(f"Horizon detected! Estimated Roll Angle: {roll_angle:.2f} degrees")

    # --- Create Visualizations ---
    debug_image = original_image.copy()
    corrected_image = rotate_and_resize(original_image, -roll_angle)

    # Draw ROI
    cv2.polylines(
        debug_image, [roi_corners], isClosed=True, color=(0, 0, 255), thickness=2
    )

    # Draw all detected Hough lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw the final calculated horizon line
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

    # --- Display Results ---
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Masked Edges within ROI", masked_edges)
    cv2.imshow("Horizon Detection (Debug)", debug_image)
    cv2.imshow("Corrected Image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- CHOOSE YOUR MODE ---
    # Set to True to run batch evaluation on the CSV file.
    # Set to False to run visual investigation on a single image.
    EVALUATION_MODE = True  # <-- CHANGE THIS TO True or False

    if EVALUATION_MODE:
        # --- BATCH EVALUATION MODE ---
        csv_path = "AI/roll3.csv"
        image_path_template = (
            "images/dataset3/distorted_images_cam1/distorted_cam1_{}.jpg"
        )

        try:
            ground_truth_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: The CSV file was not found at '{csv_path}'")
            exit()

        errors = []
        print(
            f"Starting evaluation on {len(ground_truth_df)} images from '{csv_path}'..."
        )

        for index, row in ground_truth_df.iterrows():
            image_num_str = str(int(row["image"])).zfill(4)
            ground_truth_angle = float(row["roll"])
            image_path = image_path_template.format(image_num_str)

            predicted_angle = get_roll_angle_for_image(image_path) + 0.1

            if predicted_angle is not None:
                error = abs(predicted_angle - ground_truth_angle)
                errors.append(error)
                print(
                    f"Image: {image_num_str} | Truth: {ground_truth_angle:6.2f} | Predicted: {predicted_angle:6.2f} | Error: {error:5.2f}"
                )
                print(sum(errors))
            else:
                print(f"Image: {image_num_str} | SKIPPED (could not process)")

        if errors:
            mean_absolute_error = sum(errors) / len(errors)
            print("\n--- EVALUATION COMPLETE ---")
            print(
                f"Successfully processed {len(errors)} / {len(ground_truth_df)} images."
            )
            print(f"Mean Absolute Error (MAE): {mean_absolute_error:.2f} degrees")
        else:
            print("\n--- EVALUATION FAILED ---")
            print("No images were processed successfully.")

    else:
        # --- VISUAL INVESTIGATION MODE ---
        print("Running in Visual Investigation Mode...")
        # Replace this with the path to any image you want to test
        TEST_IMAGE_PATH = (
            "images/dataset3/distorted_images_cam1/distorted_cam1_0502.jpg"
        )
        visualize_single_image(TEST_IMAGE_PATH)
