
"""
This script runs an end-to-end workflow to:
1. Create a Digital Surface Model (DSM) from a directory of LAZ files.
2. Read a CSV of camera locations and orientations.
3. For every 10th camera location, generate a 360-degree simulated skyline.
4. For each of those points, overlay the simulated skyline onto the corresponding 360-degree panoramic photograph.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import constants  

from geospatial_processing import create_dsm_from_laz, generate_simulated_skylines

LAZ_DIRECTORY = Path("DSM_Data/") 
LOCATIONS_CSV_PATH = Path(constants.GPS_LOCATION_AND_ORIENTATION)
PHOTO_DIR = Path(constants.PANORAMAS_DIR) 

# 2. Output Paths
OUTPUT_DSM_PATH = Path("./output/dsm/surface_model.tif")
SIMULATED_SKYLINE_DIR = Path("./output/simulated_skylines/")
OUTPUT_VISUALIZATION_DIR = Path("./output/visualizations/")

# 3. Processing Parameters
DSM_RESOLUTION = 1.0
OBSERVER_HEIGHT = 1.7
POINT_CRS = "EPSG:27700" # IMPORTANT: The CRS of your 'easting'/'northing' columns.

def load_skyline(skyline_path: Path) -> pd.DataFrame:
    """Loads an r.horizon output file into a pandas DataFrame."""
    if not skyline_path.exists():
        print(f"  Warning: Skyline file not found: {skyline_path}")
        return pd.DataFrame()
    skyline_df = pd.read_csv(skyline_path, delim_whitespace=True, header=None, names=['azimuth', 'horizon_angle', 'distance'])
    return skyline_df

def visualize_panoramic_skyline(image_path, skyline_df, camera_orientation_deg, output_path):
    """
    Overlays a 360-degree simulated skyline on a panoramic (equirectangular) image.
    
    Args:
        image_path: Path to the 360-degree panoramic photograph.
        skyline_df: DataFrame containing the full 360-degree simulated skyline.
        camera_orientation_deg: The compass bearing (0=N, 90=E) corresponding to the
                                horizontal center of the panorama.
        output_path: Path to save the final plot.
    """
    if not image_path.exists():
        print(f"  Error: Image not found at {image_path}, skipping.")
        return
    try:
        image = plt.imread(image_path)
    except Exception as e:
        print(f"  Error: Could not read image {image_path}. Skipping. Error: {e}")
        return

    # --- Coordinate System Mapping for Panorama ---

    # An equirectangular panorama covers 360 degrees horizontally and 180 degrees vertically.
    # Vertical range is from -90 (nadir/bottom) to +90 (zenith/top).
    vertical_angle_range = [-90, 90]

    # The horizontal center of the image corresponds to the camera's orientation.
    # We convert the compass bearing (0=N) to r.horizon azimuth (0=E).
    center_azimuth = (450 - camera_orientation_deg) % 360
    
    # The image's horizontal axis spans 360 degrees, centered on `center_azimuth`.
    azimuth_range = [center_azimuth - 180, center_azimuth + 180]

    # To ensure the skyline line wraps around the 0/360 degree boundary seamlessly,
    # we create an "unrolled" version of the skyline data that extends beyond the
    # 0-360 range. This allows matplotlib to draw a continuous line.
    skyline_extended = pd.concat([
        skyline_df.assign(azimuth=skyline_df['azimuth'] - 360),
        skyline_df,
        skyline_df.assign(azimuth=skyline_df['azimuth'] + 360)
    ], ignore_index=True)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(20, 10)) # Use a 2:1 aspect ratio for the plot

    # Display the panoramic image, setting its extent to our calculated angular boundaries.
    ax.imshow(image, extent=[*azimuth_range, *vertical_angle_range], aspect='auto')

    # Overlay the extended skyline data. Matplotlib will only plot the part within the axes' limits.
    ax.plot(
        skyline_extended['azimuth'],
        skyline_extended['horizon_angle'],
        color='cyan',
        linewidth=2,
        label='Simulated Skyline'
    )
    
    # Set the plot limits and labels
    ax.set_xlim(azimuth_range)
    ax.set_ylim(vertical_angle_range)
    ax.set_title(f"360° Panoramic Skyline Visualization\n(Image: {image_path.name})")
    ax.set_xlabel(f"Azimuth (Degrees, r.horizon system). Image Center at {center_azimuth:.1f}° (Bearing: {camera_orientation_deg}°)")
    ax.set_ylabel("Elevation Angle (Degrees from horizontal)")
    ax.legend()
    
    # Set ticks to be more intuitive if possible
    major_ticks = np.arange(azimuth_range[0], azimuth_range[1] + 1, 45)
    ax.set_xticks(major_ticks)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig) # Close the figure to free up memory


# --- MAIN WORKFLOW (No changes needed here) ---

def main():
    """Drives the entire process from DSM creation to final visualization."""
    
    print("--- Starting Panoramic Skyline Visualization Workflow ---")

    # === STEP 1: Create Digital Surface Model (DSM) ===
    print("\n[STEP 1/4] Creating Digital Surface Model from LAZ files...")
    if not create_dsm_from_laz(LAZ_DIRECTORY, OUTPUT_DSM_PATH, resolution=DSM_RESOLUTION):
        print("Workflow failed at DSM creation. Exiting.")
        return
    print("[STEP 1/4] DSM creation complete.")

    # === STEP 2: Prepare Observer Points ===
    print("\n[STEP 2/4] Preparing observer points from CSV...")
    if not LOCATIONS_CSV_PATH.exists():
        print(f"FATAL ERROR: Locations CSV file not found at: {LOCATIONS_CSV_PATH}")
        return

    locations_df = pd.read_csv(LOCATIONS_CSV_PATH)
    locations_subset = locations_df.iloc[::10].copy()
    print(f"Loaded {len(locations_df)} total locations. Sub-sampling every 10th point, for a total of {len(locations_subset)} points.")

    geometry = gpd.points_from_xy(locations_subset['easting'], locations_subset['northing'])
    grid_gdf = gpd.GeoDataFrame(locations_subset.rename(columns={'n': 'id'}), geometry=geometry, crs=POINT_CRS)
    
    temp_grid_path = OUTPUT_VISUALIZATION_DIR.parent / "temp_observer_points.gpkg"
    grid_gdf.to_file(temp_grid_path, driver="GPKG", layer="observers") # Added layer name for robustness
    print(f"Observer points prepared and saved to temporary file: {temp_grid_path}")
    
    # === STEP 3: Generate Simulated Skylines ===
    print("\n[STEP 3/4] Generating simulated skylines for each point...")
    if not generate_simulated_skylines(OUTPUT_DSM_PATH, temp_grid_path, SIMULATED_SKYLINE_DIR, observer_height=OBSERVER_HEIGHT):
        print("Workflow failed at skyline generation. Exiting.")
        return
    print("[STEP 3/4] Skyline generation complete.")
    
    # === STEP 4: Create Visualizations ===
    print("\n[STEP 4/4] Creating final panoramic visualizations...")
    OUTPUT_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    for index, row in locations_subset.iterrows():
        point_id = row['n']
        image_filename = row['filename']
        orientation = row['Orientation']
        
        print(f"  -> Visualizing point ID: {point_id}...")

        skyline_path = SIMULATED_SKYLINE_DIR / f"sim_skyline_{point_id}.txt"
        image_path = PHOTO_DIR / image_filename
        output_plot_path = OUTPUT_VISUALIZATION_DIR / f"visualization_pano_{point_id}.png"

        skyline_df = load_skyline(skyline_path)
        if skyline_df.empty:
            continue

        # Use the new panoramic visualization function
        visualize_panoramic_skyline(
            image_path=image_path,
            skyline_df=skyline_df,
            camera_orientation_deg=orientation,
            output_path=output_plot_path,
        )
        print(f"     Saved visualization to: {output_plot_path}")
    
    print("\n--- Workflow finished successfully! ---")


if __name__ == "__main__":
    main()