from pathlib import Path

# --- Project Root Directory ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data Directories ---
# Main images directory
IMAGES_DIR = PROJECT_ROOT / "images"

# COLMAP directory
COLMAP_DIR = PROJECT_ROOT / "colmap_fisheye_project"

# Path to DSM-related data
LIDAR_DATA_DIR = PROJECT_ROOT / "DSM_Data" / "cleaned_laser"
LIDAR_SKYLINE_RESOLUTION = 64  # The resolution of the skyline will become 5120 (i.e the number of pixels in the panorama) / this factor
LIDAR_OUTPUT_DIR = PROJECT_ROOT / "output" / "lidar_skyline"

OBSERVER_GRID_PATH = (
    PROJECT_ROOT / "data" / "observer_grid.shp"
)  # TODO: CREATE THIS FILE IN QGI

PANORAMA_IMG_WIDTH = 5120
PANORAMA_IMG_HEIGHT = 960

# Image dimensions (found when running split_stacked_images.py)
SPLIT_IMG_WIDTH = 1280
SPLIT_IMG_HEIGHT = 720

# Image dimensions after undistortion
OUTPUT_IMG_WIDTH = 1280
OUTPUT_IMG_HEIGHT = 960

# Distortion parameters CSV path
DISTORTION_PARAMS_CSV = PROJECT_ROOT / "distortion_parameters.csv"

# Output directories for various processing stages
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Simulated skylines and .tif file will here
GEOSPATIAL_OUTPUT_DIR = OUTPUT_DIR / "geospatial"

MEGALOC_DATABASE = PROJECT_ROOT / "megaloc_db.pkl"
