import cv2
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math

# TODO: Instead of creating one undistorted image, use the information from both cameras to create 4 90-degree views.

class FisheyeUndistorter:
    def __init__(self, csv_path=None, csv_data=None):
        if csv_path is None and csv_data is None:
            raise ValueError("Either csv_path or csv_data must be provided.")
        
        self.camera_specs = {
            'sensor': 'IMX219',
            'native_resolution': (3280, 2464),
            'input_image_resolution': (1280, 720),
            'focal_length_mm': 0.87,
            'diagonal_fov_degrees': 200
        }
        
        self._calculate_derived_parameters()

        if csv_path is not None:
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
        
        self.pixel_pitch_x_mm = sensor_width_mm / W_native
        self.pixel_pitch_y_mm = sensor_height_mm / H_native
        self.avg_pixel_pitch_mm = (self.pixel_pitch_x_mm + self.pixel_pitch_y_mm) / 2.0
        
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
        
        f_out_pixels = (output_width / 2.0) / math.tan(math.radians(output_rectilinear_fov_degrees / 2.0))
        print(f"Output rectilinear image: {output_width}x{output_height}, f_out={f_out_pixels:.2f}px for {output_rectilinear_fov_degrees} deg HFOV.")

        map_x_remap = np.zeros((output_height, output_width), dtype=np.float32)
        map_y_remap = np.zeros((output_height, output_width), dtype=np.float32)

        for v_out in range(output_height):
            for u_out in range(output_width):
                x_n_rect = (u_out - cx_out) / f_out_pixels
                y_n_rect = (v_out - cy_out) / f_out_pixels

                theta_scene_radians = math.atan(math.sqrt(x_n_rect**2 + y_n_rect**2))
                theta_scene_degrees = math.degrees(theta_scene_radians)

                real_height_mm_distorted = self.map_angle_to_distorted_radius_mm(theta_scene_degrees)
                r_d_pixels_on_input_image = real_height_mm_distorted / self.avg_pixel_pitch_mm
                
                alpha_rect = math.atan2(y_n_rect, x_n_rect)

                u_fisheye = self.cx_input + r_d_pixels_on_input_image * math.cos(alpha_rect)
                v_fisheye = self.cy_input + r_d_pixels_on_input_image * math.sin(alpha_rect)
                
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
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        undistorted = self.undistort_image(image, **kwargs)
        
        if output_path:
            cv2.imwrite(str(output_path), undistorted)
            print(f"Undistorted image saved to: {output_path}")
        
        return image, undistorted
    
    def batch_process(self, input_folder, output_folder, **kwargs):
        os.makedirs(output_folder, exist_ok=True)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in extensions):
                input_path = os.path.join(input_folder, filename)
                output_file_name = f"undistorted_{os.path.splitext(filename)[0]}.png"
                output_path = os.path.join(output_folder, output_file_name)
                
                try:
                    print(f"Processing {input_path}...")
                    self.process_image_file(input_path, output_path, **kwargs)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    def visualize_lut_mapping(self, save_path="lut_mapping_analysis.png"):
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

    def suggest_fov_for_fill_factor(self, desired_fill_factor=0.8):
        """
        Suggests an output FOV that will fill approximately the desired fraction of the output image.
        desired_fill_factor: 0.0 to 1.0, where 1.0 means fill the entire output
        """
        max_input_radius = min(self.cx_input, self.cy_input, 
                            self.camera_specs['input_image_resolution'][0] - self.cx_input,
                            self.camera_specs['input_image_resolution'][1] - self.cy_input)
        
        max_angle_deg = self.lut_y_a_deg[-1]
        
        suggested_fov = max_angle_deg * 2 * desired_fill_factor
        suggested_fov = min(suggested_fov, 170)
        
        print(f"Suggested FOV for {desired_fill_factor*100}% fill: {suggested_fov:.1f}°")
        return suggested_fov

    def save_comparison_plot(self, original, undistorted, save_path="comparison.png"):
        if original is None or undistorted is None:
            print("Cannot save comparison plot: One or both images are None.")
            return

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
    csv_file_path = "Arvid/distortion_parameters.csv" 
    
    try:
        import constants
        input_image_path_single = os.path.join(constants.CAM1_FOLDER, "cam1_0801.jpg")
    except (ImportError, AttributeError):
        print("Warning: 'constants' module or 'CAM1_FOLDER' not found. Please set 'input_image_path_single' manually.")
        input_image_path_single = "test_fisheye.jpg"
        if not os.path.exists(input_image_path_single):
            print(f"Creating dummy test image at {input_image_path_single}")
            dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Test Fisheye", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            cv2.imwrite(input_image_path_single, dummy_img)

    output_image_path_single = "undistorted_output_image.png"
    
    output_w = 1280
    output_h = 720
    output_fov = 90  # Reduced from 120 to 90 for more zoom

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        print(f"Current working directory: {os.getcwd()}")
        if os.path.exists(os.path.join("..", csv_file_path)):
            csv_file_path = os.path.join("..", csv_file_path)
            print(f"Found CSV at: {csv_file_path}")
        else:
            print("Please ensure the CSV path is correct.")
            exit(1)
            
    try:
        undistorter = FisheyeUndistorter(csv_path=csv_file_path)
    except Exception as e:
        print(f"Error initializing FisheyeUndistorter: {e}")
        exit(1)

    undistorter.visualize_lut_mapping("lut_mapping_analysis.png")
    
    if os.path.exists(input_image_path_single):
        print(f"\nProcessing single image: {input_image_path_single}")
        try:
            original_img, undistorted_img = undistorter.process_image_file(
                input_image_path_single, 
                output_image_path_single,
                output_width=output_w,
                output_height=output_h,
                output_rectilinear_fov_degrees=output_fov
            )
            undistorter.save_comparison_plot(original_img, undistorted_img, "single_image_comparison.png")
            print(f"Single image processing complete. Output: {output_image_path_single}")
        except Exception as e:
            print(f"Error processing single image {input_image_path_single}: {e}")
    else:
        print(f"Single input image not found: {input_image_path_single}. Skipping single image test.")

    print("\nScript finished.")

if __name__ == "__main__":
    main()