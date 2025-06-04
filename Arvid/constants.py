from pathlib import Path
import os

# --- Project Root Directory ---
PROJECT_ROOT_OF_SCRIPTS = Path(__file__).resolve().parent

# --- Data Directories ---
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "dataset1/images"

COLMAP_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "colmap_fisheye_project"

CAM1_FOLDER = os.path.join(COLMAP_DIR, "images_cam1")
CAM2_FOLDER = os.path.join(COLMAP_DIR, "images_cam2")

# found when running split_stacked_images.py
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720