from pathlib import Path
import os

# --- Project Root Directory ---
PROJECT_ROOT_OF_SCRIPTS = Path(__file__).resolve().parent

# --- Data Directories ---
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "images/dataset1/images"

COLMAP_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "colmap_fisheye_project"

CAM1_DISTORTED_FOLDER = os.path.join(PROJECT_ROOT_OF_SCRIPTS, "images", "cam1_distorted_images")
CAM2_DISTORTED_FOLDER = os.path.join(PROJECT_ROOT_OF_SCRIPTS, "images", "cam2_distorted_images")

# TODO: Get actual path
CAM1_UNDISTORTED_FOLDER = os.path.join(PROJECT_ROOT_OF_SCRIPTS, "images", "cam1_undistorted_images")
CAM2_UNDISTORTED_FOLDER = os.path.join(PROJECT_ROOT_OF_SCRIPTS, "images", "cam2_undistorted_images")

# found when running split_stacked_images.py
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720