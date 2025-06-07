from pathlib import Path

# --- Project Root Directory ---
PROJECT_ROOT_OF_SCRIPTS = Path(__file__).resolve().parent

# --- Data Directories ---
<<<<<<< HEAD
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "images/dataset1/images"
DATASET1_CSV = PROJECT_ROOT_OF_SCRIPTS.parent / "images/gps_interpolated_with_orientation.csv"
=======
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS / "images/dataset1/images"
DATASET1_CSV = PROJECT_ROOT_OF_SCRIPTS / "images/dataset1/gps_interpolated_with_orientation.csv"
>>>>>>> ee00199f649a4304b1d5fb64dbb4d7f6e4386d03

COLMAP_DIR = PROJECT_ROOT_OF_SCRIPTS / "colmap_fisheye_project"

# Use Path objects consistently for better cross-platform compatibility
CAM1_DISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam1_distorted_images"
CAM2_DISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam2_distorted_images"

CAM1_UNDISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam1_undistorted_images"
CAM2_UNDISTORTED_FOLDER = PROJECT_ROOT_OF_SCRIPTS / "images" / "cam2_undistorted_images"

# Image dimensions (found when running split_stacked_images.py)
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720