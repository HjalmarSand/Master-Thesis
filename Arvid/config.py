from pathlib import Path
import os

# --- Project Root Directory ---
PROJECT_ROOT_OF_SCRIPTS = Path(__file__).resolve().parent

# --- Data Directories ---
DATASET1_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "dataset1/images"

COLMAP_DIR = PROJECT_ROOT_OF_SCRIPTS.parent / "colmap_fisheye_project"