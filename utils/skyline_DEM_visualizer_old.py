"""
A script to visualize a sample of points from a .laz or .las file.

This script uses PDAL to read a point cloud and offers two methods for 3D visualization:
1. matplotlib: A common plotting library, good for basic visualization.
2. open3d: A specialized 3D library for a better, interactive experience.

INSTRUCTIONS:
1. Make sure you have the required libraries installed:
   - For both methods:
     pip install pdal numpy
   - For matplotlib (basic):
     pip install matplotlib
   - For open3d (recommended for better interaction):
     pip install open3d

2. Edit the CONFIGURATION section below.
3. Run the script from your terminal:
   python visualize_laz.py
"""

import pdal
import numpy as np
from pathlib import Path
import json  # <--- THIS IS THE FIX. The missing import is now added.

# --- CONFIGURATION ---
# --- Edit these values to match your setup ---

# 1. Path to your LAZ or LAS file.
LAZ_FILE_PATH = Path("./DSM_Data/y657_69_7500_17.laz")  # <--- CHANGE THIS to your file

# 2. Maximum number of points to load for visualization.
#    This prevents running out of memory with very large files.
#    A value between 500,000 and 2,000,000 is usually good for modern computers.
MAX_POINTS_TO_LOAD = 1_000_000

# 3. Choose your visualization library.
#    'open3d' -> (Recommended) Provides a fast, interactive window.
#    'matplotlib' -> Basic 3D plot, can be slow to interact with.
VISUALIZATION_METHOD = 'open3d'  # <--- CHANGE to 'matplotlib' if you prefer

# --- End of Configuration ---


def visualize_with_matplotlib(x, y, z):
    """Generates a 3D scatter plot using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Error: Matplotlib is not installed. Please run 'pip install matplotlib'")
        return

    print("Generating plot with Matplotlib... (this may take a moment)")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colormap to show elevation and make points small
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=0.5, marker='.')

    ax.set_title(f"Point Cloud Visualization ({len(x):,} points)")
    ax.set_xlabel("Easting (X)")
    ax.set_ylabel("Northing (Y)")
    ax.set_zlabel("Elevation (Z)")

    # Add a color bar to show what the colors mean
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('Elevation')
    
    # Set aspect ratio to be equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = x.mean()
    mid_y = y.mean()
    mid_z = z.mean()
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    print("Displaying plot. Close the plot window to exit.")
    plt.show()


def visualize_with_open3d(x, y, z):
    """Generates an interactive 3D visualization using Open3D."""
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt # Needed for colormap
    except ImportError:
        print("Error: Open3D or Matplotlib is not installed. Please run 'pip install open3d matplotlib'")
        return

    print("Preparing data for Open3D...")
    points_xyz = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        z_normalized = (z - z_min) / (z_max - z_min)
        colors = plt.cm.viridis(z_normalized)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("Displaying interactive window. Use your mouse to rotate/zoom.")
    print("Press 'Q' in the window to close.")
    o3d.visualization.draw_geometries([pcd])


def main():
    """Main function to drive the script."""
    print(f"--- Starting LAZ Visualization ---")

    if not LAZ_FILE_PATH.exists():
        print(f"FATAL ERROR: File not found at '{LAZ_FILE_PATH}'")
        return

    print(f"Reading and sampling up to {MAX_POINTS_TO_LOAD:,} points from file...")
    
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(LAZ_FILE_PATH)
            },
            {
                "type": "filters.sample",
                "count": MAX_POINTS_TO_LOAD
            }
        ]
    }

    try:
        # json.dumps() requires the 'json' module to be imported
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
    except Exception as e:
        print(f"An error occurred during PDAL processing: {e}")
        return

    points = pipeline.arrays[0]
    
    if len(points) == 0:
        print("Error: No points were read from the file.")
        return
        
    print(f"Successfully loaded {len(points):,} points.")

    x = points['X']
    y = points['Y']
    z = points['Z']

    if VISUALIZATION_METHOD.lower() == 'matplotlib':
        visualize_with_matplotlib(x, y, z)
    elif VISUALIZATION_METHOD.lower() == 'open3d':
        visualize_with_open3d(x, y, z)
    else:
        print(f"Error: Unknown visualization method '{VISUALIZATION_METHOD}'.")
        print("Please choose either 'matplotlib' or 'open3d'.")

    print("\n--- Visualization complete ---")


if __name__ == "__main__":
    main()