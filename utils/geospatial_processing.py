import pdal
import json
import subprocess
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np

def create_dsm_from_laz(laz_dir: Path, output_dsm_path: Path, resolution: float = 1.0):
    """
    Finds all .laz files in a directory, merges them, and creates a single DSM.

    Args:
        laz_dir: Path to the directory containing .laz files.
        output_dsm_path: The path to save the final DSM GeoTIFF.
        resolution: The resolution of the output DSM in meters.
    """
    print(f"Searching for .laz files in: {laz_dir}")
    laz_files = list(laz_dir.glob("*.laz"))
    if not laz_files:
        print("Error: No .laz files found in the specified directory.")
        return False
    
    print(f"Found {len(laz_files)} .laz files to process.")
    
    # Create the parent directory for the output if it doesn't exist
    output_dsm_path.parent.mkdir(parents=True, exist_ok=True)

    # This PDAL pipeline reads multiple LAZ files, merges them, and writes a single DSM
    pipeline_json = {
        "pipeline": [
            # The reader can be specified for each file
            *laz_files,
            {
                "type": "filters.merge"
            },
            {
                "type": "writers.gdal",
                "filename": str(output_dsm_path),
                "gdaldriver": "GTiff",
                "output_type": "max",  # Use the highest point (buildings, trees)
                "resolution": resolution,
                "data_type": "float32"
            }
        ]
    }
    
    try:
        print("Executing PDAL pipeline to create DSM...")
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        print(f"DSM created successfully at {output_dsm_path}")
        return True
    except Exception as e:
        print(f"An error occurred during PDAL DSM creation: {e}")
        return False

def generate_simulated_skylines(dsm_path: Path, grid_path: Path, output_dir: Path, observer_height: float = 1.7):
    """
    Generates 360-degree skyline profiles for each point in a grid file using GRASS r.horizon.
    
    NOTE: This requires GRASS to be installed and the `r.horizon` command to be in the system's PATH.
        This is usually the case if you have installed QGIS.
    """
    if not dsm_path.exists():
        print(f"Error: DSM file not found at {dsm_path}")
        return False
    if not grid_path.exists():
        print(f"Error: Observer grid file not found at {grid_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading observer grid from: {grid_path}")
    grid_gdf = gpd.read_file(grid_path)
    print(f"Found {len(grid_gdf)} observer points.")

    for index, point in grid_gdf.iterrows():
        point_id = point.get('id', index) 
        observer_x = point.geometry.x
        observer_y = point.geometry.y

        print(f"  -> Processing skyline for point ID: {point_id}...")
        
        output_skyline_file = output_dir / f"sim_skyline_{point_id}.txt"
        
        command = [
            "r.horizon",
            "-d", # Output distance to horizon as well
            f"elevation={dsm_path}",
            f"coordinates={observer_x},{observer_y}",
            f"observer_elevation={observer_height}",
            "step=1", # 1-degree steps
            f"output={output_skyline_file}"
        ]
        
        try:
            # Using capture_output=True to hide GRASS's verbose output unless there's an error
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            print("\nFATAL ERROR: `r.horizon` command not found.")
            print("Please ensure GRASS GIS is installed and its bin directory is in your system's PATH.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running r.horizon for point {point_id}:")
            print(e.stderr) # Print the error from the tool
            return False
            
    print("\nSimulated skyline generation complete.")
    return True