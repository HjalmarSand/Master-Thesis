import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
CSV_FILE_PATH = "images/dataset1/gps_interpolated_with_orientation.csv"
INPUT_IMAGE_DIR = "images/dataset1/undistorted_cam1_left"
OUTPUT_IMAGE_DIR = "output_images_corrected_from_yaw"

# --- MODEL TUNING PARAMETER ---
# This is the crucial constant that links yaw rate to roll.
# You MUST experiment with this value.
# - Start with a small value like 0.5.
# - If the correction is not enough, increase it (e.g., 1.0, 1.5).
# - If the image over-corrects (tilts the other way), decrease it.
# This value depends on your boat's speed and handling characteristics.
K_ROLL_YAW_RATE = 0.8

# --- Main Processing Logic ---


def correct_boat_tilt_from_yaw():
    """
    Infers boat roll (tilt) from its rate of turn (yaw rate) and uses this
    estimated roll to level the horizon in corresponding images.
    """
    # 1. Ensure the output directory exists
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"Created output directory: {OUTPUT_IMAGE_DIR}")

    # 2. Load the CSV data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE_PATH} was not found.")
        return

    # --- 3. DATA PRE-PROCESSING: Calculate Yaw Rate and Estimate Roll ---
    print("Calculating yaw rate and estimating roll...")

    # Calculate the difference in orientation from one row to the next
    # This is our initial yaw rate in degrees per second
    yaw_diff = df["Orientation"].diff().fillna(0)

    # Correct for the 360-degree wraparound problem
    # If the change is > 180, it went the short way (e.g., 359 -> 1 is a +2 change)
    # If the change is < -180, it went the short way (e.g., 1 -> 359 is a -2 change)
    yaw_rate = yaw_diff.apply(
        lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
    )

    # --- Apply our physical model to estimate roll ---
    # Estimated Roll = Yaw Rate * Proportionality Constant
    # This assumes the boat leans OUT of the turn.
    df["estimated_roll"] = yaw_rate * K_ROLL_YAW_RATE

    # (Optional but Recommended) Smooth the estimated roll to reduce jitter
    # A rolling window of 3-5 frames is usually good.
    df["smoothed_roll"] = (
        df["estimated_roll"].rolling(window=5, center=True, min_periods=1).mean()
    )

    print(f"Successfully loaded and processed CSV data for {len(df)} entries.")

    # --- 4. Process each image using the new estimated roll ---
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Correcting Images"):
        image_filename = "undistorted_cam1_" + str(row["n"]) + "_left" + ".jpg"

        # Use the SMOOTHED estimated roll for a more stable video
        estimated_roll = row["smoothed_roll"]

        input_path = os.path.join(INPUT_IMAGE_DIR, image_filename)

        if not os.path.exists(input_path):
            print(f"\nWarning: Image not found: {input_path}")
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"\nWarning: Could not read image: {input_path}")
            continue

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # The correction angle is the negative of the estimated roll
        correction_angle = -estimated_roll

        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        corrected_image = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        output_path = os.path.join(OUTPUT_IMAGE_DIR, f"corrected_{image_filename}")
        cv2.imwrite(output_path, corrected_image)

    print("\nProcessing complete!")
    print(f"Corrected images have been saved to the '{OUTPUT_IMAGE_DIR}' directory.")


# --- Run the script ---
if __name__ == "__main__":
    correct_boat_tilt_from_yaw()
