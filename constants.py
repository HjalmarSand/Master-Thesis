from pathlib import Path

import numpy as np

# --- Project Root Directory ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data Directories ---
# Main images directory
IMAGES_DIR = PROJECT_ROOT / "images"

# Dataset directory containing the original stacked images
DATASET1_DIR = IMAGES_DIR / "dataset1" / "images"

# COLMAP directory
COLMAP_DIR = PROJECT_ROOT / "colmap_fisheye_project"

# Path to DSM-folder
LIDAR_DATA_DIR = PROJECT_ROOT / "DSM_Data"

OBSERVER_GRID_PATH = PROJECT_ROOT / "data" / "observer_grid.shp" # TODO: CREATE THIS FILE IN QGIS

# Distorted image folders (output from splitting stacked images)
CAM1_DISTORTED_FOLDER = IMAGES_DIR / "cam1_distorted_images"
CAM2_DISTORTED_FOLDER = IMAGES_DIR / "cam2_distorted_images"

# Undistorted image folders (output from fisheye correction)
CAM1_UNDISTORTED_LEFT_FOLDER = IMAGES_DIR / "cam1_undistorted/cam1_undistorted_left_images"
CAM1_UNDISTORTED_RIGHT_FOLDER = IMAGES_DIR / "cam1_undistorted/cam1_undistorted_right_images"

CAM2_UNDISTORTED_LEFT_FOLDER = IMAGES_DIR / "cam2_undistorted/cam2_undistorted_left_images"
CAM2_UNDISTORTED_RIGHT_FOLDER = IMAGES_DIR / "cam2_undistorted/cam2_undistorted_right_images"

# Panorama output directory
PANORAMAS_DIR = IMAGES_DIR / "panoramas"

SKYLINE_MASKS_DIR = IMAGES_DIR / "skyline_masks"

# Image dimensions (found when running split_stacked_images.py)
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720

# Image dimensions after undistortion
OUTPUT_IMG_WIDTH = 1280
OUTPUT_IMG_HEIGHT = 960

# Distortion parameters CSV path
DISTORTION_PARAMS_CSV = PROJECT_ROOT / "distortion_parameters.csv"
GPS_LOCATION_AND_ORIENTATION = PROJECT_ROOT / "gps_interpolated_with_orientation.csv"

# Output directories for various processing stages
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Simulated skylines and .tif file will here
GEOSPATIAL_OUTPUT_DIR = OUTPUT_DIR / "geospatial"

MEGALOC_DATABASE = PROJECT_ROOT / "megaloc_db.pkl"