# Uses backward mapping to figure out what color a pixel on a blank canvas should have if it were taken with a rectilinear camera instead

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
from tqdm import tqdm

import constants

class FisheyeUndistorter:
    def __init__(self, csv_path=None, csv_data=None):
        if csv_path is None and csv_data is None:
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
            csv_path = Path(csv_path)
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
        self.pixel_pitch_y_mm = self.pixel_pitch_x_mm
        self.avg_pixel_pitch_mm = self.pixel_pitch_x_mm

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
                        output_rectilinear_fov_degrees=90,
                        yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0):
        H_input_img, W_input_img = fisheye_image.shape[:2]

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

        map_x_remap = np.zeros((output_height, output_width), dtype=np.float32)
        map_y_remap = np.zeros((output_height, output_width), dtype=np.float32)

        yaw_rad = math.radians(yaw_degrees)
        pitch_rad = math.radians(pitch_degrees)
        roll_rad = math.radians(roll_degrees)

        # Yaw (around Y-axis)
        R_yaw_matrix = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])

        # Pitch (around X-axis)
        R_pitch_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])

        # Roll (around Z-axis)
        R_roll_matrix = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        # TODO: Correct matrix mult order? Check
        R_rect_to_fisheye = R_roll_matrix @ R_pitch_matrix @ R_yaw_matrix

        for v_out in range(output_height):
            for u_out in range(output_width):
                ray_x_rect = (u_out - cx_out)
                ray_y_rect = (v_out - cy_out)
                ray_z_rect = f_out_pixels

                ray_vec_local_rect = np.array([ray_x_rect, ray_y_rect, ray_z_rect])

                # Normalized 3D vector for pixel for virtual camera
                ray_vec_local_rect_norm = ray_vec_local_rect / np.linalg.norm(ray_vec_local_rect)

                ray_vec_fisheye = R_rect_to_fisheye @ ray_vec_local_rect_norm

                # Step 1: For this output pixel, what real-world angle SHOULD it represent? 
                # Converting to degrees since that is used in real height polynomial fit
                theta_scene_radians = math.atan2(np.linalg.norm(ray_vec_fisheye[:2]), ray_vec_fisheye[2])
                theta_scene_degrees = math.degrees(theta_scene_radians)

                # Step 2: For that angle, where did the fisheye lens ACTUALLY put the information? (Find the landing spot in mm)
                distorted_radius_mm = self.map_angle_to_distorted_radius_mm(theta_scene_degrees)

                # Step 3: Convert that physical landing spot into a source coordinate radius
                distorted_radius_pixels = distorted_radius_mm / self.avg_pixel_pitch_mm

                alpha_fisheye = math.atan2(ray_vec_fisheye[1], ray_vec_fisheye[0])

                # Step 4: These are the exact coordinates in the source fisheye image where the color information I need is located
                u_fisheye = self.cx_input + distorted_radius_pixels * math.cos(alpha_fisheye)
                v_fisheye = self.cy_input + distorted_radius_pixels * math.sin(alpha_fisheye)

                map_x_remap[v_out, u_out] = u_fisheye
                map_y_remap[v_out, u_out] = v_fisheye

        undistorted_image = cv2.remap(fisheye_image, map_x_remap, map_y_remap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return undistorted_image

    def undistort_fisheye_to_split_views(self, fisheye_image,
                                        output_width, output_height,
                                        rectilinear_fov_degrees=90,
                                        horizontal_angle_offset_degrees=45,
                                        vertical_angle_offset_degrees=0,
                                        rotation_degrees=0): # This is the roll
        left_yaw_degrees = -horizontal_angle_offset_degrees
        right_yaw_degrees = horizontal_angle_offset_degrees

        left_view = self.undistort_image(
            fisheye_image,
            output_width=output_width,
            output_height=output_height,
            output_rectilinear_fov_degrees=rectilinear_fov_degrees,
            yaw_degrees=left_yaw_degrees, # Virtual camera yaw for left view
            pitch_degrees=vertical_angle_offset_degrees,
            roll_degrees=rotation_degrees # Virtual camera roll
        )

        right_view = self.undistort_image(
            fisheye_image,
            output_width=output_width,
            output_height=output_height,
            output_rectilinear_fov_degrees=rectilinear_fov_degrees,
            yaw_degrees=right_yaw_degrees, # Virtual camera yaw for right view
            pitch_degrees=vertical_angle_offset_degrees,
            roll_degrees=rotation_degrees # Virtual camera roll
        )
        return left_view, right_view
        
    # Method for just rolled image
    def get_roll_corrected_view(self, fisheye_image, output_width, output_height,
                                rectilinear_fov_degrees=90, roll_degrees=0.0):
        # This will be a central view with only roll applied
        return self.undistort_image(
            fisheye_image,
            output_width=output_width,
            output_height=output_height,
            output_rectilinear_fov_degrees=rectilinear_fov_degrees,
            yaw_degrees=0.0,  # No yaw
            pitch_degrees=0.0, # No pitch
            roll_degrees=roll_degrees # Only roll applied
        )

    def batch_process_images(self, input_dir, output_dir_left, output_dir_right,
                                output_width, output_height,
                                rectilinear_fov_degrees=90,
                                horizontal_angle_offset_degrees=45,
                                vertical_angle_offset_degrees=0,
                                rotation_degrees=0):

        Path(output_dir_left).mkdir(parents=True, exist_ok=True)
        Path(output_dir_right).mkdir(parents=True, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in tqdm(image_files, desc=f"Processing {Path(input_dir).name}"):
            input_path = os.path.join(input_dir, filename)
            fisheye_image = cv2.imread(input_path)

            if fisheye_image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            tqdm.write(f"Processing {filename}...") 
            
            left_view, right_view = self.undistort_fisheye_to_split_views(
                fisheye_image,
                output_width, output_height,
                rectilinear_fov_degrees=rectilinear_fov_degrees,
                horizontal_angle_offset_degrees=horizontal_angle_offset_degrees,
                vertical_angle_offset_degrees=vertical_angle_offset_degrees,
                rotation_degrees=rotation_degrees
            )

            # 1. Get the filename without extension
            original_stem, extension = os.path.splitext(filename)
            
            # 2. We replace 'distorted_' with an empty string ''.
            new_stem = original_stem.replace('distorted_', '')

            # 3. Build the new filename string.
            new_filename_left = f"undistorted_{new_stem}_left{extension}"
            new_filename_right = f"undistorted_{new_stem}_right{extension}"

            # 4. Join the output directory with the new filename.
            output_path_left = os.path.join(output_dir_left, new_filename_left)
            output_path_right = os.path.join(output_dir_right, new_filename_right)

            cv2.imwrite(output_path_left, left_view)
            cv2.imwrite(output_path_right, right_view)

        print(f"Finished processing batch for {Path(input_dir).name}.")

    def save_comparison_plot(self, original_image, left_view, right_view, center_roll_view, output_path, title="Fisheye Undistortion"):
        plt.figure(figsize=(24, 6))

        plt.subplot(1, 4, 1)
        # OpenCV reads images in BGR, matplotlib expects RGB
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Distorted Image')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(center_roll_view, cv2.COLOR_BGR2RGB))
        plt.title('Roll Corrected Only (Central)')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(left_view, cv2.COLOR_BGR2RGB))
        plt.title('Undistorted Left View')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(right_view, cv2.COLOR_BGR2RGB))
        plt.title('Undistorted Right View')
        plt.axis('off')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(output_path)
        print(f"Comparison plot saved to: {output_path}")

def main():
    input_image_path = os.path.join(constants.CAM1_DISTORTED_FOLDER, 'distorted_cam1_0894.jpg')
    output_base_dir = 'undistorted_output'
    output_dir_left = os.path.join(output_base_dir, 'left_views')
    output_dir_right = os.path.join(output_base_dir, 'right_views')
    output_plot_dir = os.path.join(output_base_dir, 'plots')

    Path(output_dir_left).mkdir(parents=True, exist_ok=True)
    Path(output_dir_right).mkdir(parents=True, exist_ok=True)
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)

    undistorter = FisheyeUndistorter()

    fisheye_image = cv2.imread(input_image_path)

    if fisheye_image is None:
        print(f"Error: Could not load image from {input_image_path}. Make sure it's in the same directory.")
        return

    print(f"Processing image: {input_image_path}")

    output_width = constants.SPLIT_IMG_WIDTH
    output_height = constants.SPLIT_IMG_HEIGHT
    rectilinear_fov_degrees = 90
    horizontal_angle_offset_degrees = 45
    vertical_angle_offset_degrees = 0
    
    # !!! IMPORTANT: Start experimenting with this value ---
    rotation_degrees = 9

    # Get the roll-corrected central view
    center_roll_view = undistorter.get_roll_corrected_view(
        fisheye_image,
        output_width,
        output_height,
        rectilinear_fov_degrees=rectilinear_fov_degrees,
        roll_degrees=rotation_degrees # Apply the determined roll
    )

    left_view, right_view = undistorter.undistort_fisheye_to_split_views(
        fisheye_image,
        output_width,
        output_height,
        rectilinear_fov_degrees=rectilinear_fov_degrees,
        horizontal_angle_offset_degrees=horizontal_angle_offset_degrees,
        vertical_angle_offset_degrees=vertical_angle_offset_degrees,
        rotation_degrees=rotation_degrees
    )

    filename_stem = Path(input_image_path).stem
    output_path_left = os.path.join(output_dir_left, f"undistorted_{filename_stem}_left.jpg")
    output_path_right = os.path.join(output_dir_right, f"undistorted_{filename_stem}_right.jpg")
    output_plot_path = os.path.join(output_plot_dir, f"comparison_plot_{filename_stem}.png")
    output_path_center_roll = os.path.join(output_plot_dir, f"undistorted_{filename_stem}_center_roll_only.jpg") # Save separately too

    cv2.imwrite(output_path_left, left_view)
    cv2.imwrite(output_path_right, right_view)
    cv2.imwrite(output_path_center_roll, center_roll_view) # Save the center roll view

    print(f"Undistortion complete. Left view saved to: {output_path_left}")
    print(f"Right view saved to: {output_path_right}")
    print(f"Center roll-corrected view saved to: {output_path_center_roll}")

    undistorter.save_comparison_plot(fisheye_image, left_view, right_view, center_roll_view, output_plot_path)

if __name__ == "__main__":
    main()