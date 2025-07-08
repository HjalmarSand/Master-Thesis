import pdal
import json
import subprocess
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
# In skyline_generator.py

import pdal
import json
from pathlib import Path

def create_dsm_from_laz(laz_dir: Path, output_dsm_path: Path, resolution: float = 1.0):
    """
    Finds all .laz files in a directory using a glob pattern, streams them,
    and creates a single DSM in a memory-efficient way.

    Args:
        laz_dir: Path to the directory containing .laz files.
        output_dsm_path: The path to save the final DSM GeoTIFF.
        resolution: The resolution of the output DSM in meters.
    """
    print(f"Searching for .laz files in: {laz_dir}")
    # Use a glob pattern for the filename. This is the key change.
    laz_pattern = laz_dir / "*.laz"
    
    # Check if any files match the pattern.
    # We can do a quick check to provide a user-friendly message.
    if not list(laz_dir.glob("*.laz")):
        print("Error: No .laz files found in the specified directory.")
        return False
    
    print("Found .laz files. Will process them as a stream.")
    
    # Create the parent directory for the output if it doesn't exist
    output_dsm_path.parent.mkdir(parents=True, exist_ok=True)

    # This updated PDAL pipeline streams data from all matching LAZ files.
    # Notice there is no `filters.merge` and only one reader entry.
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(laz_pattern) # The reader uses the wildcard pattern here
            },
            {
                "type": "writers.gdal",
                "filename": str(output_dsm_path),
                "gdaldriver": "GTiff",
                "output_type": "max",  # Use the highest point (buildings, trees)
                "resolution": resolution,
                "data_type": "float32",
                "gdalopts": "COMPRESS=LZW,TILED=YES" # Added compression and tiling for efficiency
            }
        ]
    }
    
    try:
        print("Executing memory-efficient PDAL pipeline to create DSM...")
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        # This will now process files sequentially without loading them all into memory.
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