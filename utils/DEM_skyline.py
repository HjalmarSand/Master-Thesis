import pdal
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math
import glob
from typing import List, Tuple, Optional

LAZ_DIRECTORY = Path("./DSM_Data/cleaned")  

SEARCH_RADIUS = 1500.0  # How far away to search for potential skyline points (in meters)
ANGLE_STEP = 1   # degrees - angular resolution of skyline
MIN_DISTANCE = 10.0     # meters - minimum distance to consider (avoid nearby clutter)
MAX_POINTS_PER_FILE = 5000000  # TODO: Implement this functionality to limit points loaded from each file for memory handling
N_POINTS_TO_AVERAGE = 10  # Number of highest points to average in each sector (1 = single highest point, >1 = average top N points)
# Query point
VIEWPOINT_X = 613491.715  
VIEWPOINT_Y = 6462211.464   
VIEWPOINT_Z = 1.7

def find_laz_files() -> List[Path]:
    """Find all LAZ files in the configured directory."""
    print(f"Searching for LAZ files in directory: {LAZ_DIRECTORY}")
    
    if not LAZ_DIRECTORY.exists():
        raise FileNotFoundError(f"Directory not found: {LAZ_DIRECTORY}")
    
    laz_files = []
    laz_patterns = ["*.laz", "*.LAZ"]
    for pattern in laz_patterns:
        found_files = list(LAZ_DIRECTORY.glob(pattern))
        laz_files.extend(found_files)
    
    laz_files = sorted(list(laz_files)) 
    
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in directory: {LAZ_DIRECTORY}")
    
    print(f"Found {len(laz_files)} LAZ files:")
    for file_path in laz_files:
        print(f"  {file_path}")
    
    return laz_files

def check_file_relevance_from_json(file_path: Path, viewpoint_coords: Tuple[float, float, float], 
                                search_radius: float) -> bool:
    """
    Check if a LAZ file contains any points within the search radius by reading its corresponding JSON file.
    This is much faster than loading points since it only reads the spatial bounds from the JSON.
    
    Args:
        file_path: Path to the LAZ file
        viewpoint_coords: Tuple of (x, y, z) coordinates for viewpoint
        search_radius: Maximum distance to search for skyline points
    
    Returns:
        bool: True if file might contain relevant points, False otherwise
    """
    vp_x, vp_y, vp_z = viewpoint_coords
    
    # Construct json path identical to corresponding LAZ file
    json_path = file_path.with_suffix('.json')
    
    try:
        if not json_path.exists():
            print(f"  WARNING: JSON file not found: {json_path.name}")
            # If no JSON file, assume it might be relevant to be safe
            return True
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Find bounding box coordinates from the JSON metadata

        # Extract coordinates from the geometry polygon
        coordinates = data['geometry']['coordinates'][0]
        box_coords = coordinates[:-1] #The last coordinate is a repeat of the first
        
        x_coords = [coord[0] for coord in box_coords]
        y_coords = [coord[1] for coord in box_coords]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Calculate minimum distance from viewpoint to the files bounding box
        # Clamp viewpoint to the bounding box to find closest point on box
        closest_x = max(min_x, min(vp_x, max_x))
        closest_y = max(min_y, min(vp_y, max_y))
        
        min_distance_to_box = np.sqrt((vp_x - closest_x)**2 + (vp_y - closest_y)**2)
        
        is_relevant = min_distance_to_box <= search_radius
        
        print(f"  Bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        print(f"  Min distance to box: {min_distance_to_box:.1f}m")
        
        if is_relevant:
            print(f"  ✓ File is relevant (within {search_radius}m)")
        else:
            print(f"  ✗ File too far away (> {search_radius}m)")
            
        return is_relevant
        
    except Exception as e:
        print(f"  WARNING: Could not read bounds from {json_path}: {e}")
        return True

def load_single_point_cloud(file_path: Path, viewpoint_coords: Tuple[float, float, float], 
                            search_radius: float, max_points: int = 500000) -> Optional[np.ndarray]:
    """Load point cloud data from a single LAZ file, filtering by distance from viewpoint."""
    print(f"Loading: {file_path.name}")
    
    vp_x, vp_y, vp_z = viewpoint_coords
    
    try:
        # Create spatial filter using PDAL's crop filter
        # Calculate bounding box around viewpoint
        buffer = search_radius # * 1.1  # Add 10% buffer to account for approximation
        min_x = vp_x - buffer
        max_x = vp_x + buffer
        min_y = vp_y - buffer
        max_y = vp_y + buffer
        
        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": str(file_path)
                },
                {
                    "type": "filters.crop",
                    "bounds": f"([{min_x}, {max_x}], [{min_y}, {max_y}])"
                }
                # Uncomment the next line to apply a random sample filter if needed for large files,
                #{
                #    "type": "filters.sample",
                #    "radius": 0.5
                #}
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        points = pipeline.arrays[0]
        
        if len(points) == 0:
            print(f"  No points within bounding box")
            return None
        
        # Apply precise circular distance filter
        distances = np.sqrt((points['X'] - vp_x)**2 + (points['Y'] - vp_y)**2)
        within_radius = distances <= search_radius
        filtered_points = points[within_radius]
        
        if len(filtered_points) == 0:
            print(f"  No points within {search_radius}m radius")
            return None
        
        print(f"  Loaded {len(filtered_points):,} points within {search_radius}m radius (filtered from {len(points):,})")
        return filtered_points
        
    except Exception as e:
        print(f"  ERROR loading {file_path}: {e}")
        return None

def load_multiple_point_clouds(file_paths: List[Path], viewpoint_coords: Tuple[float, float, float], 
                            search_radius: float, max_points_per_file: int = 500000) -> np.ndarray:
    """Load and combine point cloud data from multiple LAZ files with distance filtering."""
    print(f"\n=== CHECKING FILE RELEVANCE USING JSON METADATA ===")
    print(f"Viewpoint: X={viewpoint_coords[0]:.2f}, Y={viewpoint_coords[1]:.2f}, Z={viewpoint_coords[2]:.2f}")
    print(f"Search radius: {search_radius}m")
    
    # First pass: Check which files are potentially relevant using their JSON metadata
    relevant_files = []
    for file_path in file_paths:
        print(f"\nChecking bounds: {file_path.name}")
        if check_file_relevance_from_json(file_path, viewpoint_coords, search_radius):
            relevant_files.append(file_path)
    
    print(f"\n=== BOUNDS CHECK RESULTS ===")
    print(f"Files potentially relevant: {len(relevant_files)}/{len(file_paths)}")
    print(f"Files skipped (too far away): {len(file_paths) - len(relevant_files)}")
    
    if not relevant_files:
        raise ValueError("No LAZ files found within the search radius based on JSON metadata!")
    
    # Second pass: Load only the relevant files
    print(f"\n=== LOADING {len(relevant_files)} RELEVANT LAZ FILES ===")
    
    all_points = []
    total_points = 0
    successful_files = 0
    failed_files = 0
    
    for file_path in relevant_files:
        points = load_single_point_cloud(file_path, viewpoint_coords, search_radius, max_points_per_file)
        if points is not None and len(points) > 0:
            all_points.append(points)
            total_points += len(points)
            successful_files += 1
        else:
            failed_files += 1
    
    if not all_points:
        raise ValueError("No relevant point clouds were successfully loaded!")
    
    print(f"\n=== FINAL LOADING RESULTS ===")
    print(f"Files successfully loaded: {successful_files}/{len(relevant_files)}")
    print(f"Files that failed to load: {failed_files}")
    print(f"Total relevant points loaded: {total_points:,}")
    
    # Combine all point clouds
    if len(all_points) == 1:
        combined_points = all_points[0]
    else:
        combined_points = np.concatenate(all_points)
    
    print(f"Combined relevant points: {len(combined_points):,}")
    
    # Show combined coordinate ranges
    x_coords = combined_points['X']
    y_coords = combined_points['Y'] 
    z_coords = combined_points['Z']
    
    print(f"Relevant data coordinate ranges:")
    print(f"  X: {x_coords.min():.2f} to {x_coords.max():.2f}")
    print(f"  Y: {y_coords.min():.2f} to {y_coords.max():.2f}")
    print(f"  Z: {z_coords.min():.2f} to {z_coords.max():.2f}")
    
    return combined_points

def get_viewpoint_coordinates():
    """
    Get viewpoint coordinates from user input or configuration.
    This version is used early in the process before loading point clouds.
    
    Returns:
        tuple: (x, y, z) coordinates of the viewpoint
    """
    print("\n=== VIEWPOINT SELECTION (REQUIRED FOR SPATIAL FILTERING) ===")
    
    print(f"Current configured viewpoint:")
    print(f"  X: {VIEWPOINT_X}")
    print(f"  Y: {VIEWPOINT_Y}")
    print(f"  Z: {VIEWPOINT_Z}")
    print()
    
    print("Options:")
    print("1. Use configured coordinates (from VIEWPOINT_X, Y, Z)")
    print("2. Enter new coordinates manually")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                print(f"Using configured viewpoint: X={VIEWPOINT_X}, Y={VIEWPOINT_Y}, Z={VIEWPOINT_Z}")
                return VIEWPOINT_X, VIEWPOINT_Y, VIEWPOINT_Z
            
            elif choice == "2":
                print("\nEnter new viewpoint coordinates:")
                x = float(input(f"X coordinate: "))
                y = float(input(f"Y coordinate: "))
                z = float(input(f"Z coordinate: "))
                print(f"Using custom viewpoint: X={x}, Y={y}, Z={z}")
                return x, y, z
            
            else:
                print("Invalid choice. Please enter 1 or 2.")
                continue
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()


def calculate_skyline(points, viewpoint_coords, search_radius, angle_step=1.0, 
                      min_distance=10.0, avg_top_n_points=1):
    """
    Calculate skyline from a viewpoint using combined point cloud data.
    
    This version uses an Azimuth system where 0 degrees = North.
    
    Args:
        points: Combined point cloud array from multiple LAZ files.
        viewpoint_coords: Tuple of (x, y, z) coordinates for viewpoint.
        search_radius: Maximum distance to search for skyline points.
        angle_step: Angular resolution in degrees.
        min_distance: Minimum distance to consider points.
        avg_top_n_points (int): The number of highest points to average per sector.
    
    Returns:
        angles: Array of angles (0-360 degrees, 0=North)
        elevations: Array of elevation angles to highest points/regions.
        distances: Array of distances to highest points/regions.
        skyline_points: Array of actual (or averaged) skyline point coordinates.
    """
    
    print(f"\n=== CALCULATING SKYLINE (0° = North) ===")
    
    vp_x, vp_y, vp_z = viewpoint_coords
    
    print(f"Viewpoint: X={vp_x:.2f}, Y={vp_y:.2f}, Z={vp_z:.2f}")
    if avg_top_n_points > 1:
        print(f"ROBUST MODE: Averaging the top {avg_top_n_points} points in each angular sector.")
    else:
        print("STANDARD MODE: Using the single highest point in each angular sector.")
    
    rel_x = points['X'] - vp_x
    rel_y = points['Y'] - vp_y
    rel_z = points['Z'] - vp_z
    
    horizontal_dist = np.sqrt(rel_x**2 + rel_y**2)
    
    angles_rad = np.arctan2(rel_x, rel_y)
    
    angles_deg = np.degrees(angles_rad)
    angles_deg = (angles_deg + 360) % 360  # Normalize to 0-360
    
    valid_mask = (horizontal_dist <= search_radius) & (horizontal_dist >= min_distance)
    valid_points = points[valid_mask]
    valid_rel_z = rel_z[valid_mask]
    valid_horizontal_dist = horizontal_dist[valid_mask]
    valid_angles = angles_deg[valid_mask]
    
    print(f"Found {len(valid_points):,} points within valid distance range ({min_distance}m to {search_radius}m)")
    
    angle_bins = np.arange(0, 360, angle_step)
    skyline_angles = []
    skyline_elevations = []
    skyline_distances = []
    skyline_points = []
    
    print("Processing angular sectors...")
    
    for i, angle in enumerate(angle_bins):
        if i % 90 == 0:
            print(f"  Processing direction {angle:.0f}°...")
            
        angle_min = angle - angle_step / 2
        angle_max = angle + angle_step / 2
        
        if angle_min < 0:
            sector_mask = (valid_angles >= (360 + angle_min)) | (valid_angles < angle_max)
        elif angle_max >= 360:
            sector_mask = (valid_angles >= angle_min) | (valid_angles < (angle_max - 360))
        else:
            sector_mask = (valid_angles >= angle_min) & (valid_angles < angle_max)
            
        if np.any(sector_mask):
            sector_rel_z = valid_rel_z[sector_mask]
            sector_dist = valid_horizontal_dist[sector_mask]
            sector_points = valid_points[sector_mask]
            
            elevation_angles_in_sector = np.degrees(np.arctan2(sector_rel_z, sector_dist))
            
            num_in_sector = len(elevation_angles_in_sector)
            num_to_average = min(avg_top_n_points, num_in_sector)

            if num_to_average <= 1:
                highest_idx = np.argmax(elevation_angles_in_sector)
                final_elevation = elevation_angles_in_sector[highest_idx]
                final_distance = sector_dist[highest_idx]
                point = sector_points[highest_idx]
                final_point_coords = [point['X'], point['Y'], point['Z']]
            else:
                indices_of_highest_points = np.argsort(elevation_angles_in_sector)[-num_to_average:]
                final_elevation = np.mean(elevation_angles_in_sector[indices_of_highest_points])
                final_distance = np.mean(sector_dist[indices_of_highest_points])
                top_points = sector_points[indices_of_highest_points]
                final_point_coords = [np.mean(top_points['X']), np.mean(top_points['Y']), np.mean(top_points['Z'])]

            skyline_angles.append(angle)
            skyline_elevations.append(final_elevation)
            skyline_distances.append(final_distance)
            skyline_points.append(final_point_coords)
        else:
            skyline_angles.append(angle)
            skyline_elevations.append(0)
            skyline_distances.append(search_radius)
            skyline_points.append([vp_x, vp_y, vp_z])
    
    print(f"Skyline calculation complete: {len(skyline_angles)} directions processed")
    
    return (np.array(skyline_angles), 
            np.array(skyline_elevations), 
            np.array(skyline_distances),
            np.array(skyline_points))


def plot_skyline(angles, elevations, distances, viewpoint_coords, num_files):
    """
    Plot the skyline in multiple views.
    The main skyline profile is plotted in a separate, dedicated window for clarity.
    The other plots (distance, polar, stats) are in a second window.
    """
    # Remap angles from 0-360 to -180-180 for a plot centered on North (0)
    plot_angles = np.where(angles >= 180, angles - 360, angles)

    # Get the sorting order for the remapped angles
    sort_indices = np.argsort(plot_angles)
    
    # Apply this sorting to all data arrays so they stay synchronized
    plot_angles_sorted = plot_angles[sort_indices]
    plot_elevations_sorted = elevations[sort_indices]
    plot_distances_sorted = distances[sort_indices]

    # --- FIGURE 1: Detailed Skyline Profile (The old 'ax1') ---
    # Create a new figure and axes specifically for the detailed skyline
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    
    ax1.plot(plot_angles_sorted, plot_elevations_sorted, 'b-', linewidth=1.5)
    ax1.fill_between(plot_angles_sorted, plot_elevations_sorted, -15, alpha=0.3, color='skyblue')
    ax1.set_xlabel('Direction (Azimuth)', fontsize=12)
    ax1.set_ylabel('Elevation Angle (degrees)', fontsize=12)
    ax1.set_title('Detailed Skyline Profile (Centered on North) at Viewpoint '+ str(VIEWPOINT_X) + " " + str(VIEWPOINT_Y))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xlim(-180, 180)
    
    ax1_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    ax1.set_xticks(ax1_ticks)
    ax1.set_xticklabels(['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'S'])
    
    fig1.suptitle(f'360° Skyline Elevation Profile (Search Radius: {SEARCH_RADIUS}m)', fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    # FIGURE 2: Supporting Data Plots 
    fig2 = plt.figure(figsize=(12, 10))
    gs = fig2.add_gridspec(2, 2, height_ratios=[3, 2]) 

    ax2 = fig2.add_subplot(gs[0, 0])  
    ax3 = fig2.add_subplot(gs[0, 1], projection='polar')
    ax4 = fig2.add_subplot(gs[1, :])  
    
    ax2.plot(plot_angles_sorted, plot_distances_sorted, 'r-', linewidth=2)
    ax2.set_xlabel('Direction (Azimuth)')
    ax2.set_ylabel('Distance to Skyline (m)')
    ax2.set_title('Distance to Skyline Points')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)
    ax2.set_xticks(ax1_ticks) 
    ax2.set_xticklabels(['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'S'])
    
    angles_rad = np.radians(angles)
    elevation_radius = 90 - elevations
    ax3.plot(angles_rad, elevation_radius, 'g-', linewidth=2)
    ax3.fill(angles_rad, elevation_radius, alpha=0.3, color='lightgreen')
    ax3.set_title('Skyline - Polar View', y=1.1)
    ax3.set_ylim(0, 180) 
    ax3.set_yticks([0, 45, 90, 135, 180]) 
    ax3.set_yticklabels(['+90°', '+45°', '0°', '-45°', '-90°'])
    ax3.set_theta_zero_location('N')  
    ax3.set_theta_direction(-1)       
    
    ax4.axis('off') 
    stats_text = f"""
    MULTI-LAZ SKYLINE STATISTICS
    
    Source: {num_files} LAZ files
    
    Viewpoint: 
    X: {viewpoint_coords[0]:.2f}
    Y: {viewpoint_coords[1]:.2f} 
    Z: {viewpoint_coords[2]:.2f}
    
    Elevation Angles:
    Max: {elevations.max():.1f}° at {angles[elevations.argmax()]:.0f}°
    Min: {elevations.min():.1f}°
    Mean: {elevations.mean():.1f}°
    
    Distances:
    Max: {distances.max():.1f} m
    Min: {distances.min():.1f} m
    Mean: {distances.mean():.1f} m
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    fig2.suptitle('Skyline Analysis - Supporting Data', fontsize=16)
    fig2.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    plt.show()


def main():
    print("MULTI-LAZ SKYLINE GENERATOR \n")
    
    try:
        laz_files = find_laz_files()
        
        print(f"\nPreparing to process {len(laz_files)} LAZ files...")
        viewpoint_coords = get_viewpoint_coordinates()
        combined_points = load_multiple_point_clouds(laz_files, viewpoint_coords, SEARCH_RADIUS, MAX_POINTS_PER_FILE)

        print(f"\nCalculating skyline with {ANGLE_STEP}° resolution...")
        
        angles, elevations, distances, skyline_points = calculate_skyline(
            combined_points, 
            viewpoint_coords, 
            SEARCH_RADIUS, 
            ANGLE_STEP, 
            MIN_DISTANCE,
            avg_top_n_points=N_POINTS_TO_AVERAGE 
        )
        
        print(f"Skyline calculated: {len(angles)} directions")
        
        plot_skyline(angles, elevations, distances, viewpoint_coords, len(laz_files))

        print(f"\n=== MULTI-LAZ SKYLINE GENERATION COMPLETE ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()