import os
import glob
import cv2
import pandas as pd
import numpy as np
import argparse

def load_distortion_lut(csv_path):
    """
    Loads the distortion parameters from the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        # Filter out extreme/invalid values for stability.
        df_valid = df[(df['ref_height'] > 0) & (df['ref_height'] < 1e9)].copy()
        df_valid.sort_values('ref_height', inplace=True)
        
        ideal_radii = df_valid['ref_height'].values
        distorted_radii = df_valid['y_a'].values
        
        print(f"Successfully loaded and prepared LUT from {csv_path}")
        return ideal_radii, distorted_radii
        
    except FileNotFoundError:
        print(f"Error: Lookup table file not found at '{csv_path}'")
        return None, None
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return None, None

def create_polar_to_cartesian_map(input_shape, output_shape, ideal_radii_lut, distorted_radii_lut):
    """
    Creates the remap grids to convert the unwrapped polar image back to cartesian.
    This version includes the correct scaling from physical units (mm) to pixels.
    """
    in_height, in_width = input_shape
    out_height, out_width = output_shape
    center_x, center_y = out_width / 2, out_height / 2

    # Create coordinate grids for the TARGET (cartesian) image.
    u, v = np.meshgrid(np.arange(out_width, dtype=np.float32), np.arange(out_height, dtype=np.float32))

    # For each pixel in the output, calculate its ideal polar coordinates (radius and angle).
    ideal_radius_mm = np.sqrt((u - center_x)**2 + (v - center_y)**2)
    theta = np.arctan2(v - center_y, u - center_x)

    # --- THIS IS THE CORRECTED LOGIC ---

    # 1. Calculate the source X coordinate (map_x): Angle -> Horizontal Pixel
    # Map theta from [-pi, pi] to the input image width [0, in_width].
    map_x = (theta + np.pi) / (2 * np.pi) * (in_width - 1)

    # 2. Calculate the source Y coordinate (map_y): Radius -> Vertical Pixel
    # Step A: Find the corresponding distorted radius in physical units (mm) using the LUT.
    distorted_radius_mm = np.interp(ideal_radius_mm, ideal_radii_lut, distorted_radii_lut)
    
    # Step B: Find the maximum radius in our LUT. This represents the outer edge.
    max_lut_radius = distorted_radii_lut.max()
    if max_lut_radius <= 0:
        raise ValueError("Maximum radius in LUT is not positive. Check your CSV data.")

    # Step C: Scale the physical radius to the pixel height of the input image.
    # This converts the 'mm' value to a 'pixel' value for the Y-axis.
    map_y = (distorted_radius_mm / max_lut_radius) * (in_height - 1)

    # 3. Ensure data types are correct for OpenCV
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    print("Generated polar-to-cartesian map successfully with correct scaling.")
    return map_x, map_y

def main():
    parser = argparse.ArgumentParser(
        description="Re-wraps a polar-unwrapped image back to a cartesian view using a distortion LUT."
    )
    parser.add_argument("--input-dir", default="images/cam1_distorted_images")
    parser.add_argument("--output-dir", default="images/cam1_undistorted_images")
    parser.add_argument("--lut-file", default="distortion_parameters.csv")
    parser.add_argument("--out-size", type=int, default=1000, help="Size of the output square image (e.g., 1000x1000).")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files.")
    args = parser.parse_args()

    ideal_radii_lut, distorted_radii_lut = load_distortion_lut(args.lut_file)
    if ideal_radii_lut is None: return

    image_paths = glob.glob(os.path.join(args.input_dir, "*.jpg")) + glob.glob(os.path.join(args.input_dir, "*.png"))
    if not image_paths:
        print(f"Error: No .jpg or .png images found in '{args.input_dir}'")
        return

    first_image = cv2.imread(image_paths[0])
    input_shape = first_image.shape[:2]
    output_shape = (args.out_size, args.out_size)
    print(f"Detected input size: {input_shape[1]}x{input_shape[0]}")
    print(f"Will generate output of size: {output_shape[1]}x{output_shape[0]}")
    
    map_x, map_y = create_polar_to_cartesian_map(input_shape, output_shape, ideal_radii_lut, distorted_radii_lut)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nStarting image processing...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, filename)

        if os.path.exists(output_path) and not args.force:
            continue

        print(f"Processing {filename}...")
        source_img = cv2.imread(img_path)
        if source_img is None: continue

        rewrapped_img = cv2.remap(
            source_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        cv2.imwrite(output_path, rewrapped_img)

    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()