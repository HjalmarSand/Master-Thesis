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
ANGLE_STEP = 1       # degrees - angular resolution of skyline
MIN_DISTANCE = 10.0     # meters - minimum distance to consider (avoid nearby clutter)
MAX_POINTS_PER_FILE = 5000000  # TODO: Adjust this based on available memory

# Query point
VIEWPOINT_X = 613417  
VIEWPOINT_Y = 6462100   
VIEWPOINT_Z = 1.5   

def find_laz_files() -> List[Path]:
    """Find all LAZ files in the configured directory."""
    print(f"Searching for LAZ files in directory: {LAZ_DIRECTORY}")
    
    if not LAZ_DIRECTORY.exists():
        raise FileNotFoundError(f"Directory not found: {LAZ_DIRECTORY}")
    
    # Finds all LAZ files in the directory
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
        
        # Calculate distance to closest point on bounding box
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
        buffer = search_radius * 1.1  # Add 10% buffer to account for approximation
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
                },
                {
                    "type": "filters.head",
                    "count": max_points
                }
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

def get_viewpoint_coordinates_early():
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

def get_viewpoint_coordinates(points):
    """
    Display information about the loaded point cloud data.
    Since viewpoint is already selected, this just shows the data ranges.
    
    Args:
        points: Combined point cloud array (used to show coordinate ranges)
    
    Returns:
        tuple: (x, y, z) coordinates of the viewpoint (already selected)
    """
    print("\n=== LOADED POINT CLOUD SUMMARY ===")
    
    # Show coordinate ranges of the loaded (filtered) data
    x_coords = points['X']
    y_coords = points['Y'] 
    z_coords = points['Z']
    
    print(f"Loaded point cloud coordinate ranges (within search radius):")
    print(f"  X: {x_coords.min():.2f} to {x_coords.max():.2f}")
    print(f"  Y: {y_coords.min():.2f} to {y_coords.max():.2f}")
    print(f"  Z: {z_coords.min():.2f} to {z_coords.max():.2f}")
    print(f"  Total points: {len(points):,}")
    
    # Return the same viewpoint coordinates that were used for filtering
    # (This maintains the same interface but viewpoint is already determined)
    return VIEWPOINT_X, VIEWPOINT_Y, VIEWPOINT_Z

def calculate_skyline(points, viewpoint_coords, search_radius, angle_step=1.0, min_distance=10.0):
    """
    Calculate skyline from a viewpoint using combined point cloud data.
    
    Args:
        points: Combined point cloud array from multiple LAZ files
        viewpoint_coords: Tuple of (x, y, z) coordinates for viewpoint
        search_radius: Maximum distance to search for skyline points
        angle_step: Angular resolution in degrees
        min_distance: Minimum distance to consider points
    
    Returns:
        angles: Array of angles (0-360 degrees)
        elevations: Array of elevation angles to highest points
        distances: Array of distances to highest points
        skyline_points: Array of actual skyline point coordinates
    """
    
    print(f"\n=== CALCULATING SKYLINE ===")
    
    # Get viewpoint coordinates
    vp_x, vp_y, vp_z = viewpoint_coords
    
    print(f"Viewpoint: X={vp_x:.2f}, Y={vp_y:.2f}, Z={vp_z:.2f}")
    print(f"Using {len(points):,} total points from all LAZ files")
    
    # Calculate relative positions of all points
    rel_x = points['X'] - vp_x
    rel_y = points['Y'] - vp_y
    rel_z = points['Z'] - vp_z
    
    # Calculate horizontal distances and angles
    horizontal_dist = np.sqrt(rel_x**2 + rel_y**2)
    angles_rad = np.arctan2(rel_y, rel_x)
    angles_deg = np.degrees(angles_rad)
    angles_deg = (angles_deg + 360) % 360  # Normalize to 0-360
    
    # Filter points within search radius and minimum distance
    valid_mask = (horizontal_dist <= search_radius) & (horizontal_dist >= min_distance)
    valid_points = points[valid_mask]
    valid_rel_x = rel_x[valid_mask]
    valid_rel_y = rel_y[valid_mask]
    valid_rel_z = rel_z[valid_mask]
    valid_horizontal_dist = horizontal_dist[valid_mask]
    valid_angles = angles_deg[valid_mask]
    
    print(f"Found {len(valid_points):,} points within search radius ({search_radius}m)")
    
    # Create angle bins
    angle_bins = np.arange(0, 360, angle_step)
    skyline_angles = []
    skyline_elevations = []
    skyline_distances = []
    skyline_points = []
    
    print("Processing angular sectors...")
    
    for i, angle in enumerate(angle_bins):
        if i % 90 == 0:  # Progress indicator
            print(f"  Processing direction {angle:.0f}°...")
            
        # Find points in this angular sector
        angle_min = angle - angle_step/2
        angle_max = angle + angle_step/2
        
        # Handle wraparound at 0/360 degrees
        if angle_min < 0:
            sector_mask = (valid_angles >= (360 + angle_min)) | (valid_angles < angle_max)
        elif angle_max >= 360:
            sector_mask = (valid_angles >= angle_min) | (valid_angles < (angle_max - 360))
        else:
            sector_mask = (valid_angles >= angle_min) & (valid_angles < angle_max)
        
        if np.any(sector_mask):
            # Find the highest point in this sector
            sector_z = valid_rel_z[sector_mask]
            sector_dist = valid_horizontal_dist[sector_mask]
            sector_points = valid_points[sector_mask]
            
            # Highest point
            highest_idx = np.argmax(sector_z)
            highest_z = sector_z[highest_idx]
            highest_dist = sector_dist[highest_idx]
            highest_point = sector_points[highest_idx]
            
            elevation_angle = np.degrees(np.arctan2(highest_z, highest_dist))
            
            skyline_angles.append(angle)
            skyline_elevations.append(elevation_angle)
            skyline_distances.append(highest_dist)
            skyline_points.append([highest_point['X'], highest_point['Y'], highest_point['Z']])
        else:
            # No points in this direction
            skyline_angles.append(angle)
            skyline_elevations.append(-90)  # Looking down at horizon
            skyline_distances.append(search_radius)
            skyline_points.append([vp_x, vp_y, vp_z])
    
    print(f"Skyline calculation complete: {len(skyline_angles)} directions processed")
    
    return (np.array(skyline_angles), 
            np.array(skyline_elevations), 
            np.array(skyline_distances),
            np.array(skyline_points))

def plot_skyline(angles, elevations, distances, viewpoint_coords, num_files):
    """Plot the skyline in multiple views."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Elevation angle vs direction (main skyline plot)
    ax1.plot(angles, elevations, 'b-', linewidth=2)
    ax1.fill_between(angles, elevations, -90, alpha=0.3, color='skyblue')
    ax1.set_xlabel('Direction (degrees)')
    ax1.set_ylabel('Elevation Angle (degrees)')
    ax1.set_title('Skyline Profile - Elevation vs Direction')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 45))
    
    # 2. Distance vs direction
    ax2.plot(angles, distances, 'r-', linewidth=2)
    ax2.set_xlabel('Direction (degrees)')
    ax2.set_ylabel('Distance to Skyline (m)')
    ax2.set_title('Distance to Skyline Points')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    ax2.set_xticks(np.arange(0, 361, 45))
    
    # 3. Polar plot of skyline
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    # Convert angles to radians and elevation to radius (90° = center, -90° = edge)
    angles_rad = np.radians(angles)
    # Map elevation: 90° -> 0 (center), -90° -> 1 (edge)
    elevation_radius = (90 - elevations) / 180
    ax3.plot(angles_rad, elevation_radius, 'g-', linewidth=2)
    ax3.fill(angles_rad, elevation_radius, alpha=0.3, color='lightgreen')
    ax3.set_title('Skyline - Polar View\n(Center=90°, Edge=-90°)')
    ax3.set_ylim(0, 1)
    ax3.set_theta_zero_location('E')  # 0° at East
    ax3.set_theta_direction(-1)  # Clockwise
    
    # 4. Statistics
    ax4.axis('off')
    stats_text = f"""
    MULTI-LAZ SKYLINE STATISTICS
    
    Source: {num_files} LAZ files
    
    Viewpoint: 
    X: {viewpoint_coords[0]:.2f}
    Y: {viewpoint_coords[1]:.2f} 
    Z: {viewpoint_coords[2]:.2f}
    
    Elevation Angles:
    Max: {elevations.max():.1f}°
    Min: {elevations.min():.1f}°
    Mean: {elevations.mean():.1f}°
    
    Distances:
    Max: {distances.max():.1f}m
    Min: {distances.min():.1f}m
    Mean: {distances.mean():.1f}m
    
    Directions with highest points:
    {angles[elevations.argmax()]:.0f}° ({elevations.max():.1f}°)
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.suptitle(f'360° Multi-LAZ Skyline Analysis (Search Radius: {SEARCH_RADIUS}m)', 
                 fontsize=16, y=0.98)
    plt.show()

def save_skyline_data(angles, elevations, distances, skyline_points, viewpoint_coords, num_files):
    """Save skyline data to CSV file."""
    filename = f"multi_laz_skyline_data_{num_files}files_vp_{viewpoint_coords[0]:.0f}_{viewpoint_coords[1]:.0f}.csv"
    
    with open(filename, 'w') as f:
        f.write("# Multi-LAZ Skyline Data\n")
        f.write(f"# Generated from {num_files} LAZ files\n")
        f.write(f"# Viewpoint: X={viewpoint_coords[0]:.2f}, Y={viewpoint_coords[1]:.2f}, Z={viewpoint_coords[2]:.2f}\n")
        f.write(f"# Search radius: {SEARCH_RADIUS}m\n")
        f.write("Direction_deg,Elevation_deg,Distance_m,Skyline_X,Skyline_Y,Skyline_Z\n")
        for i in range(len(angles)):
            f.write(f"{angles[i]:.1f},{elevations[i]:.3f},{distances[i]:.3f},"
                   f"{skyline_points[i][0]:.3f},{skyline_points[i][1]:.3f},{skyline_points[i][2]:.3f}\n")
    
    print(f"Skyline data saved to: {filename}")

def main():
    """Main function to generate skyline from multiple LAZ files."""
    print("=== MULTI-LAZ SKYLINE GENERATOR ===\n")
    
    try:
        # Find LAZ files to process
        laz_files = find_laz_files()
        
        print(f"\nPreparing to process {len(laz_files)} LAZ files...")
        
        # Get viewpoint coordinates from user first (needed for spatial filtering)
        print("\nNote: Viewpoint coordinates are needed early for spatial filtering optimization")
        viewpoint_coords = get_viewpoint_coordinates_early()
        
        # Load and combine all point clouds with spatial filtering
        combined_points = load_multiple_point_clouds(laz_files, viewpoint_coords, SEARCH_RADIUS, MAX_POINTS_PER_FILE)
        
        if len(combined_points) == 0:
            print("No relevant points found within search radius!")
            return
        
        # Calculate skyline using combined point cloud
        print(f"\nCalculating skyline with {ANGLE_STEP}° resolution...")
        angles, elevations, distances, skyline_points = calculate_skyline(
            combined_points, viewpoint_coords, SEARCH_RADIUS, ANGLE_STEP, MIN_DISTANCE
        )
        
        print(f"Skyline calculated: {len(angles)} directions")
        
        # Plot results
        plot_skyline(angles, elevations, distances, viewpoint_coords, len(laz_files))
        
        # Save data
        save_skyline_data(angles, elevations, distances, skyline_points, viewpoint_coords, len(laz_files))
        
        print(f"\n=== MULTI-LAZ SKYLINE GENERATION COMPLETE ===")
        print(f"Processed {len(laz_files)} LAZ files successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()