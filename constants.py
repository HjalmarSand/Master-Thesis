from pathlib import Path

# --- Project Root Directory ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data Directories ---
# Main images directory
IMAGES_DIR = PROJECT_ROOT / "images"

# Dataset directory containing the original stacked images
DATASET1_DIR = IMAGES_DIR / "dataset1" / "images"

# COLMAP directory
COLMAP_DIR = PROJECT_ROOT / "colmap_fisheye_project"

# Distorted image folders (output from splitting stacked images)
CAM1_DISTORTED_FOLDER = IMAGES_DIR / "cam1_distorted_images"
CAM2_DISTORTED_FOLDER = IMAGES_DIR / "cam2_distorted_images"

# Undistorted image folders (output from fisheye correction)
CAM1_UNDISTORTED_FOLDER = IMAGES_DIR / "cam1_undistorted_images"
CAM2_UNDISTORTED_FOLDER = IMAGES_DIR / "cam2_undistorted_images"

# Image dimensions (found when running split_stacked_images.py)
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720

# Distortion parameters CSV path
DISTORTION_PARAMS_CSV = PROJECT_ROOT / "distortion_parameters.csv"

# Output directories for various processing stages
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"