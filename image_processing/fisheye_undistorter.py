# Uses backward mapping to figure out what color a pixel on a blank canvas should have if it were taken with a rectilinear camera instead

# Uses backward mapping to figure out what color a pixel on a blank canvas should have if it were taken with a rectilinear camera instead

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy import interpolate

matplotlib.use("Agg")
import math
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import constants


# It contains the logic from original 'for' loop.
def _process_single_image_wrapper(args):
    """
    Wrapper function to process a single image.
    It unpacks arguments to be used with pool.map.
    """
    # Unpack all arguments
    (
        undistorter_instance,
        input_path,
        output_dir_left,
        output_dir_right,
        output_width,
        output_height,
        rectilinear_fov_degrees,
        horizontal_angle_offset_degrees,
        vertical_angle_offset_degrees,
        rotation_degrees,
        horizon_bend_correction,
    ) = args

    filename = os.path.basename(input_path)

    try:
        fisheye_image = cv2.imread(str(input_path))
        if fisheye_image is None:
            return (f"Warning: Could not read image {filename}. Skipping.", filename)

        # Call the instance method to perform the core logic
        left_view, right_view = undistorter_instance.undistort_fisheye_to_split_views(
            fisheye_image,
            output_width,
            output_height,
            rectilinear_fov_degrees,
            horizontal_angle_offset_degrees,
            vertical_angle_offset_degrees,
            rotation_degrees,
            horizon_bend_correction,
        )

        original_stem, extension = os.path.splitext(filename)
        new_stem = original_stem.replace("distorted_", "")
        new_filename_left = f"undistorted_{new_stem}_left{extension}"
        new_filename_right = f"undistorted_{new_stem}_right{extension}"
        output_path_left = os.path.join(output_dir_left, new_filename_left)
        output_path_right = os.path.join(output_dir_right, new_filename_right)

        cv2.imwrite(output_path_left, left_view)
        cv2.imwrite(output_path_right, right_view)

        return (None, filename)
    except Exception as e:
        return (f"Error processing {filename}: {e}", filename)


class FisheyeUndistorter:
    @staticmethod
    def calculate_pitch_correction_from_offset(pixel_offset, output_width, fov_degrees):
        """
        Converts a vertical pixel offset into a camera pitch angle in degrees.
        This is a static method as it does not depend on a class instance.
        """
        if output_width == 0 or fov_degrees <= 0 or fov_degrees >= 180:
            return 0.0

        # Calculate the focal length in pixels for the virtual rectilinear camera
        fov_radians = math.radians(fov_degrees / 2.0)
        focal_length_pixels = (output_width / 2.0) / math.tan(fov_radians)

        if focal_length_pixels == 0:
            return 0.0

        # The angle is the arctangent of the opposite (offset) over adjacent (focal length)
        pitch_radians = math.atan(pixel_offset / focal_length_pixels)
        return math.degrees(pitch_radians)

    def __init__(self, csv_path=None, csv_data=None):
        if csv_path is None and csv_data is None:
            csv_path = constants.DISTORTION_PARAMS_CSV

        if csv_path is None and csv_data is None:
            raise ValueError("Either csv_path or csv_data must be provided.")

        self.camera_specs = {
            "sensor": "IMX219",
            "native_resolution": (3280, 2464),
            "input_image_resolution": (
                constants.SPLIT_IMG_WIDTH,
                constants.SPLIT_IMG_HEIGHT,
            ),
            "focal_length_mm": 0.87,
            "diagonal_fov_degrees": 200,
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

        required_columns = ["y_a", "real_height", "ref_height", "optical_distortion"]
        missing_columns = [
            col
            for col in required_columns
            if col not in self.distortion_data_df.columns
        ]
        if missing_columns:
            raise ValueError(f"CSV data missing required columns: {missing_columns}")

        for col in ["optical_distortion"]:
            if self.distortion_data_df[col].dtype == "object":
                self.distortion_data_df[col] = pd.to_numeric(
                    self.distortion_data_df[col], errors="coerce"
                )

        print(f"Loaded {len(self.distortion_data_df)} distortion calibration points.")
        self._setup_interpolation()

    def _calculate_derived_parameters(self):
        W_native, H_native = self.camera_specs["native_resolution"]
        W_input, H_input = self.camera_specs["input_image_resolution"]

        # NOTE: This value was changed from 3.674 to the value below, which has resulted in vastly better normalization.
        # Previously the actual sensor width from the datasheet was used, but now instead the sensor width is derived
        # from checking the maximum real height in the distortion_parameters.csv and realising that the sensor isnt fully used.
        # Instead the entire circular fisheye image is contained in the sensor, with black areas in all corners, where the sensor
        # is not getting information. Using the actual physical dimensions of the sensor, when it is not all in use, gave the wrong
        # conclusion of pixel pitch when the distance from the center to the edge corresponding 100 degrees fov is actually
        # 1.449585mm according to the csv-file.
        sensor_width_mm = 2.89917

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

        print(f"Using sensor width: {sensor_width_mm:.3f}mm derived form CSV file")
        print(
            f"Calculated pixel pitch x: {self.pixel_pitch_x_mm * 1000:.3f}um, y: {self.pixel_pitch_y_mm * 1000:.3f}um (avg: {self.avg_pixel_pitch_mm * 1000:.3f}um)"
        )

        self.cx_input = W_input / 2.0
        self.cy_input = H_input / 2.0
        print(
            f"Assuming input image is a CROP. Using principal point ({self.cx_input}, {self.cy_input}) for the {W_input}x{H_input} image."
        )

    def _setup_interpolation(self):
        valid_data = (
            self.distortion_data_df[["y_a", "real_height"]]
            .dropna()
            .sort_values(by="y_a")
        )

        self.lut_y_a_deg = valid_data["y_a"].values
        self.lut_real_height_mm = valid_data["real_height"].values

        if len(self.lut_y_a_deg) < 2:
            raise ValueError(
                "Not enough valid data points in LUT for interpolation (need at least 2)."
            )

        self.map_angle_to_distorted_radius_mm = interpolate.interp1d(
            self.lut_y_a_deg,
            self.lut_real_height_mm,
            kind="linear",
            bounds_error=False,
            fill_value=(self.lut_real_height_mm[0], self.lut_real_height_mm[-1]),
        )
        print("Interpolator 'map_angle_to_distorted_radius_mm' created.")

    def undistort_image(
        self,
        fisheye_image,
        output_width=None,
        output_height=None,
        output_rectilinear_fov_degrees=90,
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        roll_degrees=0.0,
        horizon_bend_correction=0.0,
    ):
        H_input_img, W_input_img = fisheye_image.shape[:2]

        if output_width is None:
            output_width = W_input_img
        if output_height is None:
            output_height = H_input_img

        cx_out = output_width / 2.0
        cy_out = output_height / 2.0

        f_out_pixels = (output_width / 2.0) / np.tan(
            np.radians(output_rectilinear_fov_degrees / 2.0)
        )

        # 1. Create a grid of output pixel coordinates (u, v)
        u_out, v_out = np.meshgrid(np.arange(output_width), np.arange(output_height))

        # 2. Calculate ray vectors for all pixels at once
        ray_x_rect = u_out - cx_out
        ray_y_rect = v_out - cy_out
        ray_z_rect = np.full_like(
            ray_x_rect, f_out_pixels
        )  # Create a Z-array of the same shape

        # Stack into a (H, W, 3) array of vectors
        ray_vectors_rect = np.stack([ray_x_rect, ray_y_rect, ray_z_rect], axis=-1)

        # 3. Normalize all ray vectors
        norms = np.linalg.norm(ray_vectors_rect, axis=-1, keepdims=True)
        ray_vectors_rect_norm = ray_vectors_rect / norms

        # 4. Build rotation matrix
        yaw_rad, pitch_rad, roll_rad = np.radians(
            [yaw_degrees, pitch_degrees, roll_degrees]
        )
        # Yaw (around Y-axis)
        R_yaw = np.array(
            [
                [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                [0, 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
            ]
        )
        # Pitch (around X-axis)
        R_pitch = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch_rad), np.sin(pitch_rad)],
                [0, -np.sin(pitch_rad), np.cos(pitch_rad)],
            ]
        )

        # Roll (around Z-axis)
        R_roll = np.array(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # R_rect_to_fisheye = R_roll @ R_pitch @ R_yaw
        R_rect_to_fisheye = R_roll @ R_yaw @ R_pitch

        # 5. Rotate all vectors at once using Einstein summation for efficiency
        # This is equivalent to R_rect_to_fisheye @ ray_vectors_rect_norm for each pixel
        ray_vectors_fisheye = np.einsum(
            "ij,hwj->hwi", R_rect_to_fisheye, ray_vectors_rect_norm
        )

        # 6. Calculate spherical coordinates for all pixels
        # Note: np.arctan2 takes (y, x) arguments
        theta_scene_radians = np.arctan2(
            np.linalg.norm(ray_vectors_fisheye[..., :2], axis=-1),
            ray_vectors_fisheye[..., 2],
        )
        alpha_fisheye = np.arctan2(
            ray_vectors_fisheye[..., 1], ray_vectors_fisheye[..., 0]
        )

        theta_scene_degrees = np.degrees(theta_scene_radians)

        # 7. Interpolate and convert to pixel radii for all pixels
        distorted_radius_mm = self.map_angle_to_distorted_radius_mm(theta_scene_degrees)
        distorted_radius_pixels = distorted_radius_mm / self.avg_pixel_pitch_mm

        # 8. Calculate the source fisheye coordinates (u_remap, v_remap) for all pixels
        map_x_remap = self.cx_input + distorted_radius_pixels * np.cos(alpha_fisheye)
        map_y_remap = self.cy_input + distorted_radius_pixels * np.sin(alpha_fisheye)

        if horizon_bend_correction != 0.0:
            # Calculate the absolute horizontal angle for each pixel in the output view
            # This correctly finds the angle relative to the fisheye's true center.
            abs_yaw_rad = np.arctan2(ray_x_rect, f_out_pixels) + yaw_rad

            # Apply a vertical shift using a cosine function.
            # This applies maximum correction at the center (0 deg) and zero at the sides (+/- 90 deg).
            # A POSITIVE correction value will "push down" the center to fix the /\ peak.
            y_shift = horizon_bend_correction * np.cos(abs_yaw_rad)

            # Subtract the shift from the vertical map to sample from a HIGHER
            # pixel coordinate (smaller Y value) in the source image. This has
            # the effect of pushing the final horizon DOWN.
            map_y_remap = map_y_remap - y_shift

        # Ensure maps are float32 for cv2.remap
        map_x_remap = map_x_remap.astype(np.float32)
        map_y_remap = map_y_remap.astype(np.float32)

        # 9. Perform the remapping
        undistorted_image = cv2.remap(
            fisheye_image,
            map_x_remap,
            map_y_remap,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return undistorted_image

    def undistort_fisheye_to_split_views(
        self,
        fisheye_image,
        output_width,
        output_height,
        rectilinear_fov_degrees=90,
        horizontal_angle_offset_degrees=45,
        vertical_angle_offset_degrees=0,
        rotation_degrees=0,
        horizon_bend_correction=0.0,
    ):  # This is the roll
        left_yaw_degrees = -horizontal_angle_offset_degrees
        right_yaw_degrees = horizontal_angle_offset_degrees

        left_view = self.undistort_image(
            fisheye_image,
            output_width=output_width,
            output_height=output_height,
            output_rectilinear_fov_degrees=rectilinear_fov_degrees,
            yaw_degrees=left_yaw_degrees,  # Virtual camera yaw for left view
            pitch_degrees=vertical_angle_offset_degrees,
            roll_degrees=rotation_degrees,  # Virtual camera roll
            horizon_bend_correction=horizon_bend_correction,
        )

        right_view = self.undistort_image(
            fisheye_image,
            output_width=output_width,
            output_height=output_height,
            output_rectilinear_fov_degrees=rectilinear_fov_degrees,
            yaw_degrees=right_yaw_degrees,  # Virtual camera yaw for right view
            pitch_degrees=vertical_angle_offset_degrees,
            roll_degrees=rotation_degrees,  # Virtual camera roll
            horizon_bend_correction=horizon_bend_correction,
        )
        return left_view, right_view

    def batch_process_images(
        self,
        input_dir,
        output_dir_left,
        output_dir_right,
        output_width,
        output_height,
        rectilinear_fov_degrees=90,
        horizontal_angle_offset_degrees=45,
        vertical_angle_offset_degrees=0.0,
        rotation_degrees=0,
        horizon_bend_correction=0.0,
    ):
        Path(output_dir_left).mkdir(parents=True, exist_ok=True)
        Path(output_dir_right).mkdir(parents=True, exist_ok=True)

        image_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Prepare arguments for each worker process
        # We need to pass the instance 'self' to the worker
        args_list = [
            (
                self,
                path,
                output_dir_left,
                output_dir_right,
                output_width,
                output_height,
                rectilinear_fov_degrees,
                horizontal_angle_offset_degrees,
                vertical_angle_offset_degrees,
                rotation_degrees,
                horizon_bend_correction,
            )
            for path in image_files
        ]

        # Use about 80% of CPU cores to leave some for system responsiveness
        num_processes = max(1, int(cpu_count() * 0.8))
        print(f"Starting parallel processing with {num_processes} cores...")

        # Create a pool of workers and distribute the tasks
        with Pool(processes=num_processes) as pool:
            # Use tqdm to create a progress bar for the parallel processing
            results = list(
                tqdm(
                    pool.imap_unordered(_process_single_image_wrapper, args_list),
                    total=len(image_files),
                    desc=f"Processing {Path(input_dir).name}",
                )
            )

        # Log any errors that occurred in the child processes
        for error, filename in results:
            if error:
                print(error)

        print(f"Finished processing batch for {Path(input_dir).name}.")

    def save_comparison_plot(
        self,
        original_image,
        left_view,
        right_view,
        center_roll_view,
        output_path,
        title="Fisheye Undistortion",
    ):
        plt.figure(figsize=(24, 6))

        plt.subplot(1, 4, 1)
        # OpenCV reads images in BGR, matplotlib expects RGB
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Distorted Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(center_roll_view, cv2.COLOR_BGR2RGB))
        plt.title("Roll Corrected Only (Central)")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(left_view, cv2.COLOR_BGR2RGB))
        plt.title("Undistorted Left View")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(right_view, cv2.COLOR_BGR2RGB))
        plt.title("Undistorted Right View")
        plt.axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent title overlap
        plt.savefig(output_path)
        print(f"Comparison plot saved to: {output_path}")
