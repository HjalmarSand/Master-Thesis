# Uses backward mapping to figure out what color a pixel on a blank canvas should have if it were taken with a rectilinear camera instead

import cv2
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path

import constants

class FisheyeUndistorter:
    def __init__(self, csv_path=None, csv_data=None):
        if csv_path is None and csv_data is None:
            # Use default CSV path from constants
            csv_path = constants.DISTORTION_PARAMS_CSV
            
        if csv_path is None and csv_data is None:
            raise ValueError("Either csv_path or csv_data must be provided.")
        
        self.camera_specs = {
            'sensor': 'IMX219',
            'native_resolution': (3280, 2464),
            'input_image_resolution': (constants.SPLIT_IMG_WIDTH, constants.SPLIT_IMG_HEIGHT),
            'focal_length_mm': 0.87,
            'diagonal_fov_degrees': 200
        }
        
        self._calculate_derived_parameters()

        if csv_path is not None:
            csv_path = Path(csv_path)  # Ensure it's a Path object
            try:
                self.distortion_data_df = pd.read_csv(csv_path)
                print(f"Loaded distortion data from: {csv_path}")
            except Exception as e:
                raise ValueError(f"Error loading CSV file {csv_path}: {e}")
        else:
            self.distortion_data_df = csv_data
        
        required_columns = ['y_a', 'real_height', 'ref_height', 'optical_distortion']
        missing_columns = [col for col in required_columns if col not in self.distortion_data_df.columns]
        if missing_columns:
            raise ValueError(f"CSV data missing required columns: {missing_columns}")
        
        for col in ['optical_distortion']:
            if self.distortion_data_df[col].dtype == 'object':
                self.distortion_data_df[col] = pd.to_numeric(self.distortion_data_df[col], errors='coerce')

        print(f"Loaded {len(self.distortion_data_df)} distortion calibration points.")
        self._setup_interpolation()

    def _calculate_derived_parameters(self):
        W_native, H_native = self.camera_specs['native_resolution']
        W_input, H_input = self.camera_specs['input_image_resolution']
        
        sensor_width_mm = 3.674
        sensor_height_mm = 2.760
        
        # !!! Pixel pitch here, used when we decide what pixel to sample by dividing height where light hits the sensor by pixel size to gain
        # !!! how many pixels from the center away the light is registered. 
        # self.pixel_pitch_x_mm = sensor_width_mm / W_native
        # self.pixel_pitch_y_mm = sensor_height_mm / H_native
        # self.avg_pixel_pitch_mm = (self.pixel_pitch_x_mm + self.pixel_pitch_y_mm) / 2.0

        # !!! The images we have are not in the same height and width ratio as the camera resolution online. I will assume that the pixel pitch
        # !!! is still equal as with the native resolution and that we have square pixels. 
        # !!! I will use the pixel pitch from the width
        self.pixel_pitch_x_mm = sensor_width_mm / W_input
        self.pixel_pitch_y_mm = self.pixel_pitch_x_mm # Equal to width pitch, although with less pixels on height this can lead to sampling outside of canvas
        self.avg_pixel_pitch_mm = self.pixel_pitch_x_mm # Same as above

        print(f"Using known IMX219 sensor width: {sensor_width_mm:.3f}mm, height: {sensor_height_mm:.3f}mm")
        print(f"Calculated pixel pitch x: {self.pixel_pitch_x_mm*1000:.3f}um, y: {self.pixel_pitch_y_mm*1000:.3f}um (avg: {self.avg_pixel_pitch_mm*1000:.3f}um)")

        self.cx_input = W_input / 2.0
        self.cy_input = H_input / 2.0
        print(f"Assuming input image is a CROP. Using principal point ({self.cx_input}, {self.cy_input}) for the {W_input}x{H_input} image.")

    def _setup_interpolation(self):
        valid_data = self.distortion_data_df[['y_a', 'real_height']].dropna().sort_values(by='y_a')
        
        self.lut_y_a_deg = valid_data['y_a'].values
        self.lut_real_height_mm = valid_data['real_height'].values

        if len(self.lut_y_a_deg) < 2:
            raise ValueError("Not enough valid data points in LUT for interpolation (need at least 2).")

        self.map_angle_to_distorted_radius_mm = interpolate.interp1d(
            self.lut_y_a_deg,
            self.lut_real_height_mm,
            kind='linear',
            bounds_error=False,
            fill_value=(self.lut_real_height_mm[0], self.lut_real_height_mm[-1])
        )
        print("Interpolator 'map_angle_to_distorted_radius_mm' created.")

    def undistort_image(self, fisheye_image, 
                        output_width=None, output_height=None, 
                        output_rectilinear_fov_degrees=90):
        H_input_img, W_input_img = fisheye_image.shape[:2]
        
        if (W_input_img, H_input_img) != self.camera_specs['input_image_resolution']:
            print(f"Warning: Input image for undistortion has dimensions ({W_input_img}x{H_input_img}), "
                f"but class was initialized for ({self.camera_specs['input_image_resolution'][0]}x"
                f"{self.camera_specs['input_image_resolution'][1]}). Scaling factors might be incorrect.")

        if output_width is None:
            output_width = W_input_img
        if output_height is None:
            output_height = H_input_img

        cx_out = output_width / 2.0
        cy_out = output_height / 2.0

        if output_rectilinear_fov_degrees <= 0 or output_rectilinear_fov_degrees >= 180:
            raise ValueError("Output rectilinear FOV must be between 0 and 180 degrees.")
        
        # Focal length. Use half output width as opposite side, then take half FOV angle. Tan angle = opposite / nearby --> nearby = opposite / tan angle
        f_out_pixels = (output_width / 2.0) / math.tan(math.radians(output_rectilinear_fov_degrees / 2.0))
        print(f"Output rectilinear image: {output_width}x{output_height}, f_out={f_out_pixels:.2f}px for {output_rectilinear_fov_degrees} deg HFOV.")

        map_x_remap = np.zeros((output_height, output_width), dtype=np.float32)
        map_y_remap = np.zeros((output_height, output_width), dtype=np.float32)

        for v_out in range(output_height):
            for u_out in range(output_width):

                # Normalize coordinates by focal length and center around cx and cy in origo
                x_n_rect = (u_out - cx_out) / f_out_pixels
                y_n_rect = (v_out - cy_out) / f_out_pixels

                # Step 1: For this output pixel, what real-world angle SHOULD it represent? 
                # Converting to degrees since that is used in real height polynomial fit
                theta_scene_radians = math.atan(math.sqrt(x_n_rect**2 + y_n_rect**2))
                theta_scene_degrees = math.degrees(theta_scene_radians)

                # Step 2: For that angle, where did the fisheye lens ACTUALLY put the information? (Find the landing spot in mm)
                real_height_mm_distorted = self.map_angle_to_distorted_radius_mm(theta_scene_degrees)

                # Step 3: Convert that physical landing spot into a source coordinate radius
                # TODO: Right now pixel pitch is calculated using native resolution, but this should be wrong since the actual images have 1280x720 res
                radius_pixels_on_input_image = real_height_mm_distorted / self.avg_pixel_pitch_mm
                
                alpha_rect = math.atan2(y_n_rect, x_n_rect)

                # Step 4: These are the exact coordinates in the source fisheye image where the color information I need is located
                u_fisheye = self.cx_input + radius_pixels_on_input_image * math.cos(alpha_rect)
                v_fisheye = self.cy_input + radius_pixels_on_input_image * math.sin(alpha_rect)
                
                map_x_remap[v_out, u_out] = u_fisheye
                map_y_remap[v_out, u_out] = v_fisheye
        
        undistorted_rectilinear_image = cv2.remap(
            fisheye_image,
            map_x_remap,
            map_y_remap,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return undistorted_rectilinear_image

    def process_image_file(self, input_path, output_path=None, **kwargs):
        input_path = Path(input_path)
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        undistorted = self.undistort_image(image, **kwargs)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), undistorted)
            print(f"Undistorted image saved to: {output_path}")
        
        return image, undistorted
    
    def batch_process(self, input_folder, output_folder, **kwargs):
        """
        Processes a whole folder of images.
        Passes all keyword arguments directly to process_image_file for each image.
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        processed = 0
        for file_path in input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                stem = file_path.stem
                if stem.startswith("distorted_"):
                    stem = stem[10:]
                output_file_name = f"undistorted_{stem}.png"
                output_path = output_folder / output_file_name
                
                try:
                    print(f"Processing {file_path}...")
                    self.process_image_file(file_path, output_path, **kwargs)
                    processed += 1
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
        
        print(f"Batch processing complete. Processed {processed} images.")
        return processed

    def visualize_lut_mapping(self, save_path=None):
        if save_path is None:
            save_path = constants.PLOTS_DIR / "lut_mapping_analysis.png"
        else:
            save_path = Path(save_path)
        
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        plt.figure(figsize=(8, 6))
        plt.plot(self.lut_y_a_deg, self.lut_real_height_mm, 'b-', label='LUT: Angle to Distorted Radius (mm)')
        plt.xlabel('Incoming Angle (y_a degrees)')
        plt.ylabel('Distorted Radial Distance on Sensor (mm)')
        plt.title('Fisheye Lens: Angle to Distorted Radius Mapping (LUT)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"LUT mapping analysis plot saved to: {save_path}")

    def save_comparison_plot(self, original, undistorted, save_path=None):
        if original is None or undistorted is None:
            print("Cannot save comparison plot: One or both images are None.")
            return

        if save_path is None:
            save_path = constants.PLOTS_DIR / "comparison.png"
        else:
            save_path = Path(save_path)
        
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Fisheye ({original.shape[1]}x{original.shape[0]})')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        plt.title(f'Undistorted Rectilinear ({undistorted.shape[1]}x{undistorted.shape[0]})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {save_path}")


def main():
    """Main function to run fisheye undistortion on cam1 and cam2 images."""
    print("Starting fisheye undistortion process...")
    
    # Initialize undistorter
    try:
        undistorter = FisheyeUndistorter()
    except Exception as e:
        print(f"Error initializing FisheyeUndistorter: {e}")
        return 1
    
    # Create visualization
    undistorter.visualize_lut_mapping()
    
    # Process cam1 images
    if constants.CAM1_DISTORTED_FOLDER.exists():
        print(f"\nProcessing cam1 images from: {constants.CAM1_DISTORTED_FOLDER}")
        processed = undistorter.batch_process(
            constants.CAM1_DISTORTED_FOLDER,
            constants.CAM1_UNDISTORTED_FOLDER,
            output_width=constants.OUTPUT_IMG_WIDTH,
            output_height=constants.OUTPUT_IMG_HEIGHT,
            output_rectilinear_fov_degrees=90
        )
        print(f"Processed {processed} cam1 images")
    else:
        print(f"Warning: cam1 distorted folder not found: {constants.CAM1_DISTORTED_FOLDER}")
    
    # Process cam2 images
    if constants.CAM2_DISTORTED_FOLDER.exists():
        print(f"\nProcessing cam2 images from: {constants.CAM2_DISTORTED_FOLDER}")
        processed = undistorter.batch_process(
            constants.CAM2_DISTORTED_FOLDER,
            constants.CAM2_UNDISTORTED_FOLDER,
            output_width=constants.OUTPUT_IMG_WIDTH,
            output_height=constants.OUTPUT_IMG_HEIGHT,
            output_rectilinear_fov_degrees=90
        )
        print(f"Processed {processed} cam2 images")
    else:
        print(f"Warning: cam2 distorted folder not found: {constants.CAM2_DISTORTED_FOLDER}")
    
    # Process a single example if available
    example_images = list(constants.CAM1_DISTORTED_FOLDER.glob("*.jpg"))
    if example_images:
        print(f"\nProcessing example image for comparison plot...")
        example_path = example_images[0]
        try:
            original_img, undistorted_img = undistorter.process_image_file(
                example_path,
                output_width=constants.OUTPUT_IMG_WIDTH,
                output_height=constants.OUTPUT_IMG_HEIGHT,
                output_rectilinear_fov_degrees=90
            )
            undistorter.save_comparison_plot(original_img, undistorted_img)
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    
    print("\nFisheye undistortion complete!")
    return 0


if __name__ == "__main__":
    exit(main())