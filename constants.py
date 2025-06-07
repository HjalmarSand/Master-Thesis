from pathlib import Path

# --- Project Root Directory ---
PROJECT_ROOT_OF_SCRIPTS = Path(__file__).resolve().parent

# --- Data Directories ---
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS / "images/dataset1/images"
DATASET1_CSV = PROJECT_ROOT_OF_SCRIPTS / "images/gps_interpolated_with_orientation.csv"

COLMAP_DIR = PROJECT_ROOT_OF_SCRIPTS / "colmap_fisheye_project"

# Use Path objects consistently for better cross-platform compatibility
CAM1_DISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam1_distorted_images"
CAM2_DISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam2_distorted_images"

CAM1_UNDISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam1_undistorted_images"
CAM2_UNDISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam2_undistorted_images"

# Image dimensions (found when running split_stacked_images.py)
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720