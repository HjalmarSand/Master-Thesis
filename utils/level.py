import cv2
import numpy as np
import math
import os

# Short script to remove tilt/roll from images based on horizon points.

FRONT_PANO_PATH = "images/cam1_undistorted_images/undistorted_cam1_1100.png"
BACK_PANO_PATH = "images/cam2_undistorted_images/undistorted_cam2_1100.png"

# Horizon points for each camera
HORIZON_POINTS = {
    FRONT_PANO_PATH: ((308, 296),(833, 380)),
    BACK_PANO_PATH: ((486, 401),(833, 344))
}
OUTPUT_DIR = "output/leveled_images"

def correct_roll(image_path, horizon_points):
    """
    Corrects the roll of an image so the horizon becomes level.

    Args:
        image_path (str): The path to the input image.
        horizon_points (tuple): A tuple containing two points (x, y) on the horizon.

    Returns:
        tuple: A tuple containing the original and the corrected image as numpy arrays.
               Returns (None, None) if the image cannot be loaded.
    """
    print(f"--- Processing image: {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # Calculate the angle of the horizon
    p1, p2 = horizon_points
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    print(angle_deg)
    
    print(f"Detected horizon angle: {angle_deg:.2f} degrees")

    # Rotate the image to level the horizon
    # To make the horizon horizontal, we need to rotate by the *negative* of its current angle.
    # We'll rotate around the center of the image.
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix. Positive values correspond to counter-clockwise rotation.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Perform the affine warp (the actual rotation)
    # The new black areas created by the rotation will be filled with black color
    corrected_img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

    return img, corrected_img

def main():
    """
    Main function to process all configured images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to '{OUTPUT_DIR}/' directory.\n")

    for img_path, points in HORIZON_POINTS.items():
        # Check if the input file exists
        if not os.path.exists(img_path):
            print(f"Error: Input file not found at {img_path}. Skipping.")
            continue
            
        original_img, corrected_img = correct_roll(img_path, points)

        if corrected_img is not None:
            # Save the corrected image
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(OUTPUT_DIR, f"{name}_leveled{ext}")
            cv2.imwrite(output_path, corrected_img)
            print(f"Corrected image saved to: {output_path}\n")

            # Display the result (original vs corrected)
            # We can draw the detected horizon line on the original for visualization
            cv2.line(original_img, points[0], points[1], (0, 0, 255), 2) # Draw red line
            
            # Combine images for side-by-side comparison
            comparison_image = np.hstack((original_img, corrected_img))
            
            # Show the comparison
            window_title = f"Roll Correction for {basename}"
            cv2.imshow(window_title, comparison_image)
            
    # Wait for a key press to close all image windows
    print("Press any key to close all image windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()