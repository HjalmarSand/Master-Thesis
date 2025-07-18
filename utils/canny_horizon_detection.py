import itertools
import math
import time

import cv2
import numpy as np
import pandas as pd


def calculate_weighted_horizon_angle(lines, params):
    """
    Calculates the weighted horizon angle.
    Now accepts a 'params' dictionary for HORIZON_ANGLE_RANGE.
    """
    if lines is None:
        return 0.0

    horizon_angle_range = params["HORIZON_ANGLE_RANGE"]

    candidate_lines = []
    for line_segment in lines:
        x1, y1, x2, y2 = line_segment[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if horizon_angle_range[0] < angle < horizon_angle_range[1]:
            candidate_lines.append({"angle": angle, "length": length})

    if not candidate_lines:
        return 0.0

    weighted_angle_sum = sum(line["angle"] * line["length"] for line in candidate_lines)
    total_length = sum(line["length"] for line in candidate_lines)
    return weighted_angle_sum / total_length if total_length > 0 else 0.0


def get_roll_angle_for_image(image_path, params):
    """
    Streamlined function that now accepts a 'params' dictionary
    containing all the tuning parameters for a single run.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        return None

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, params["BLUR_KERNEL_SIZE"], 0)

    edges = cv2.Canny(
        blurred_image, params["CANNY_THRESHOLD_1"], params["CANNY_THRESHOLD_2"]
    )

    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi_top_y = int(h * params["ROI_TOP_IGNORE_PERCENT"])
    roi_bottom_y = int(h * (1 - params["ROI_BOTTOM_IGNORE_PERCENT"]))
    roi_left_x = int(w * params["ROI_LEFT_IGNORE_PERCENT"])
    roi_right_x = int(w * (1 - params["ROI_RIGHT_IGNORE_PERCENT"]))
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
        threshold=params["HOUGH_THRESHOLD"],
        minLineLength=params["HOUGH_MIN_LINE_LENGTH"],
        maxLineGap=params["HOUGH_MAX_LINE_GAP"],
    )

    return calculate_weighted_horizon_angle(lines, params)


if __name__ == "__main__":
    param_grid = {
        "CANNY_THRESHOLD_1": [50, 75],
        "CANNY_THRESHOLD_2": [150],
        "BLUR_KERNEL_SIZE": [(3, 3), (5, 5), (7, 7)],  # Must be a tuple of odd numbers
        "HOUGH_THRESHOLD": [30, 50, 70],
        "HOUGH_MIN_LINE_LENGTH": [30, 50, 70],
        "HOUGH_MAX_LINE_GAP": [10, 20],
        "ROI_TOP_IGNORE_PERCENT": [0.40],
        "ROI_BOTTOM_IGNORE_PERCENT": [0.45],
        "ROI_LEFT_IGNORE_PERCENT": [0.4],
        "ROI_RIGHT_IGNORE_PERCENT": [0.4],
        "HORIZON_ANGLE_RANGE": [(-15.0, 15.0)],
    }

    csv_path = "AI/roll3.csv"
    image_path_template = "images/dataset3/distorted_images_cam1/distorted_cam1_{}.jpg"

    # Create all possible combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_combinations = len(param_combinations)
    print(f"--- Starting Grid Search ---")
    print(f"Found {total_combinations} parameter combinations to test.")

    try:
        ground_truth_df = pd.read_csv(csv_path)
        print(f"Loaded {len(ground_truth_df)} ground truth entries from '{csv_path}'.")
    except FileNotFoundError:
        print(f"FATAL: The CSV file was not found at '{csv_path}'")
        exit()

    grid_search_results = []
    start_time = time.time()

    for i, params in enumerate(param_combinations):
        print(f"\n[{i + 1}/{total_combinations}] Testing combination: {params}")

        errors = []
        for index, row in ground_truth_df.iterrows():
            image_num_str = str(int(row["image"])).zfill(4)
            ground_truth_angle = float(row["roll"])
            image_path = image_path_template.format(image_num_str)

            predicted_angle = get_roll_angle_for_image(image_path, params)

            if predicted_angle is not None:
                error = abs(predicted_angle - ground_truth_angle)
                errors.append(error)

        # Calculate MAE for this parameter combination
        if errors:
            mean_absolute_error = sum(errors) / len(errors)
            print(f"--> MAE for this combination: {mean_absolute_error:.4f} degrees")
            grid_search_results.append({"mae": mean_absolute_error, "params": params})
        else:
            print("--> No images could be processed with this combination. Skipping.")

    end_time = time.time()
    print(f"\n--- Grid Search Finished in {end_time - start_time:.2f} seconds ---")

    if grid_search_results:
        # Sort results by error in ascending order
        sorted_results = sorted(grid_search_results, key=lambda x: x["mae"])

        best_result = sorted_results[0]

        print("\n--- Best Performing Parameters ---")
        print(f"Lowest Mean Absolute Error: {best_result['mae']:.4f} degrees")
        print("Parameters:")
        for key, value in best_result["params"].items():
            print(f"  {key}: {value}")

        print("\n--- Top 5 Results ---")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i + 1}. MAE: {result['mae']:.4f} | Params: {result['params']}")
    else:
        print("Grid search completed, but no valid results were found.")
