import pdal
import numpy as np
from pathlib import Path
import json  


# 1. Path to your LAZ or LAS file.
LAZ_FILE_PATH = Path("./DSM_Data/y657_69_7500_17.laz")  

# 2. Maximum number of points to load for visualization.
MAX_POINTS_TO_LOAD = 1_000

#    'open3d' -> (Recommended) Provides a fast, interactive window.
#    'matplotlib' -> Basic 3D plot, can be slow to interact with.
VISUALIZATION_METHOD = 'matplotlib'  



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
            "radius": 1.0  # or another appropriate radius value
        }
    ]
    }

    try:
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