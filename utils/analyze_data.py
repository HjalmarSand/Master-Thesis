import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import constants
import glob 
import re

<<<<<<< HEAD
GPS_CSV_PATH = "images/data_2326/gps_interpolated.csv"
=======

GPS_CSV_PATH = constants.GPS_LOCATION_AND_ORIENTATION  
>>>>>>> 9c696b9654e69d79fc9122013e16de2906d2bd3e
OUTPUT_DIR = "trajectory_analysis_output" # where to save plots

def calculate_distance_easting_northing(e1, n1, e2, n2):
    """Calculates distance between two Easting/Northing points."""
    return math.sqrt((e2 - e1)**2 + (n2 - n1)**2)

def calculate_angle_difference(angle1, angle2):
    """Calculates the shortest difference between two angles (0-360 degrees)."""
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff

def extract_numeric_from_filename(filename):
    """Extracts the numeric part from filenames like '0801.jpg' for sorting."""
    try:
        return int(os.path.splitext(filename)[0])
    except ValueError:
        return np.nan 

def analyze_trajectory(csv_path):
    """
    Analyzes trajectory data from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    required_cols = ['filename', 'easting', 'northing', 'Orientation']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {', '.join(required_cols)}")
        print(f"Found columns: {', '.join(df.columns)}")
        return

    # Convert coordinates to numeric, coercing errors to NaN
    df['easting'] = pd.to_numeric(df['easting'], errors='coerce')
    df['northing'] = pd.to_numeric(df['northing'], errors='coerce')
    df['Orientation'] = pd.to_numeric(df['Orientation'], errors='coerce')

    df.dropna(subset=['easting', 'northing', 'Orientation'], inplace=True)

    if df.empty:
        print("Error: No valid data after cleaning NaNs.")
        return

    df['sort_key'] = df['filename'].apply(extract_numeric_from_filename)
    df.dropna(subset=['sort_key'], inplace=True) 
    df.sort_values(by='sort_key', inplace=True)
    df.drop(columns=['sort_key'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 2:
        print("Not enough data points ( < 2) to analyze trajectory.")
        return

    print(f"Successfully loaded and sorted {len(df)} data points.")

    distances = []
    orientation_changes = []

    for i in range(1, len(df)):
        dist = calculate_distance_easting_northing(
            df['easting'].iloc[i-1], df['northing'].iloc[i-1],
            df['easting'].iloc[i], df['northing'].iloc[i]
        )
        distances.append(dist)

        # Orientation change
        orient_change = calculate_angle_difference(
            df['Orientation'].iloc[i-1], df['Orientation'].iloc[i]
        )
        orientation_changes.append(orient_change)

    df['distance_to_prev_m'] = pd.Series([np.nan] + distances) # First point has no prev distance
    df['orientation_change_deg'] = pd.Series([np.nan] + orientation_changes)

    print("\n--- Trajectory Statistics ---")
    total_images = len(df)
    total_distance_m = sum(distances)
    avg_distance_per_segment_m = np.mean(distances) if distances else 0
    median_distance_per_segment_m = np.median(distances) if distances else 0
    max_distance_per_segment_m = np.max(distances) if distances else 0
    min_distance_per_segment_m = np.min(distances) if distances else 0

    avg_orientation_change_deg = np.mean(np.abs(orientation_changes)) if orientation_changes else 0
    median_orientation_change_deg = np.median(np.abs(orientation_changes)) if orientation_changes else 0
    max_abs_orientation_change_deg = np.max(np.abs(orientation_changes)) if orientation_changes else 0

    print(f"Total number of images processed: {total_images}")
    print(f"Total trajectory length: {total_distance_m:.2f} meters")
    print(f"Average distance between consecutive images: {avg_distance_per_segment_m:.2f} m")
    print(f"Median distance between consecutive images: {median_distance_per_segment_m:.2f} m")
    print(f"Max distance between consecutive images: {max_distance_per_segment_m:.2f} m")
    print(f"Min distance between consecutive images: {min_distance_per_segment_m:.2f} m")
    print(f"Average absolute orientation change: {avg_orientation_change_deg:.2f} degrees")
    print(f"Median absolute orientation change: {median_orientation_change_deg:.2f} degrees")
    print(f"Max absolute orientation change: {max_abs_orientation_change_deg:.2f} degrees")

    # Bounding box
    min_easting, max_easting = df['easting'].min(), df['easting'].max()
    min_northing, max_northing = df['northing'].min(), df['northing'].max()
    print(f"Bounding Box (Easting): {min_easting:.2f} to {max_easting:.2f} (Width: {max_easting - min_easting:.2f} m)")
    print(f"Bounding Box (Northing): {min_northing:.2f} to {max_northing:.2f} (Height: {max_northing - min_northing:.2f} m)")


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Plotting the trajectory 
    plt.figure(figsize=(10, 8))
    plt.plot(df['easting'], df['northing'], marker='o', linestyle='-', markersize=3, label='Trajectory')
    plt.plot(df['easting'].iloc[0], df['northing'].iloc[0], 'go', markersize=8, label='Start') # Start point
    plt.plot(df['easting'].iloc[-1], df['northing'].iloc[-1], 'ro', markersize=8, label='End') # End point
    plt.title('Trajectory Plot')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'trajectory_plot.png'))
    print(f"\nTrajectory plot saved to {os.path.join(OUTPUT_DIR, 'trajectory_plot.png')}")
    # plt.show() # Uncomment to display plot interactively

    orientation_scale = avg_distance_per_segment_m * 0.8 if avg_distance_per_segment_m > 0 else 1.0

    # Quiver expects angle in radians, 0 East, counter-clockwise, or U,V components
    angles_rad = np.deg2rad(90 - df['Orientation'])
    U = orientation_scale * np.cos(angles_rad) # Eastward component
    V = orientation_scale * np.sin(angles_rad) # Northward component

    plt.figure(figsize=(12, 10))
    plt.plot(df['easting'], df['northing'], linestyle='-', color='lightblue', alpha=0.7, label='_nolegend_')
    plt.quiver(df['easting'], df['northing'], U, V,
        angles='xy', scale_units='xy', scale=1,
        width=0.003, headwidth=3, headlength=4, color='blue',
        label='Orientation')
    plt.plot(df['easting'].iloc[0], df['northing'].iloc[0], 'go', markersize=8, label='Start')
    plt.plot(df['easting'].iloc[-1], df['northing'].iloc[-1], 'ro', markersize=8, label='End')
    plt.title('Trajectory with Orientation Arrows')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'trajectory_orientation_plot.png'))
    print(f"Trajectory with orientation plot saved to {os.path.join(OUTPUT_DIR, 'trajectory_orientation_plot.png')}")


    print("\nAnalysis complete. Plots saved in", OUTPUT_DIR)

def analyze_and_plot_results(directories=None, directory="."):
    """
    Finds all 'output*.csv' files in one or more directories, calculates adjusted average error,
    and plots the results.
    
    Args:
        directories: List of directory paths or dictionary with {label: path} pairs
                    If None, uses the single 'directory' parameter
        directory: Single directory path (used if directories is None)
    """

    ERROR_ADJUSTMENT = 9.30 # Average length between two GPS points in the dataset
    CSV_FILE_PATTERN = "output*.csv"  # Pattern to match relevant CSV files
    
    # Handle different input formats
    if directories is None:
        # Single directory mode (backward compatibility)
        directories = {"Results": directory}
    elif isinstance(directories, list):
        # Convert list to dictionary with automatic labels
        directories = {f"Folder {i+1}": path for i, path in enumerate(directories)}
    elif not isinstance(directories, dict):
        print("Error: 'directories' must be a list of paths or a dictionary of {label: path}")
        return
    
    # Define colors for different curves
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_data_found = False
    
    # Process each directory
    for idx, (label, dir_path) in enumerate(directories.items()):
        print(f"\nProcessing directory: {label} ({dir_path})")
        
        # Discover all relevant CSV files in the specified directory
        search_path = os.path.join(dir_path, CSV_FILE_PATTERN)
        csv_files = glob.glob(search_path)
        
        if not csv_files:
            print(f"  Warning: No files found matching the pattern '{CSV_FILE_PATTERN}' in {dir_path}")
            continue
        
        print(f"  Found {len(csv_files)} result files to analyze.")
        
        plot_data = []
        
        # Process each file to extract radius and calculate adjusted error
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            
            # Use a regular expression to reliably extract the numeric radius from the filename
            match = re.search(r'output(\d+)\.csv', filename)
            if not match:
                print(f"  Warning: Skipping file with unexpected name format: {filename}")
                continue
                
            radius = int(match.group(1))
            
            try:
                df = pd.read_csv(filepath)
                
                # Calculate the average error
                average_error = df['distance_meters'].mean()
                
                # Since the average error doesnt take the radius exclusion into account, we adjust the average error
                adjusted_average_error = average_error - ERROR_ADJUSTMENT * (radius + 1) 
                
                plot_data.append({
                    'radius': radius,
                    'adjusted_error': adjusted_average_error
                })
                
                print(f"    - Processed {filename}: Radius={radius}, Adjusted Avg Error={adjusted_average_error:.2f}m")
                
            except pd.errors.EmptyDataError:
                print(f"  Warning: The file {filename} is empty. Skipping.")
            except Exception as e:
                print(f"  An error occurred while processing {filename}: {e}")
        
        if not plot_data:
            print(f"  No data could be processed for {label}.")
            continue
        
        all_data_found = True
        
        # Sort the data by radius to ensure the plot is drawn in a logical order
        plot_data.sort(key=lambda item: item['radius'])
        
        y_values = [item['adjusted_error'] for item in plot_data]
        x_values = [item['radius'] * 9.30 for item in plot_data]
        
        # Choose color and marker
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot the curve for this directory
        ax.plot(x_values, y_values, 
                marker=marker, 
                linestyle='-', 
                color=color, 
                label=label,
                markersize=8,
                linewidth=2)
    
    if not all_data_found:
        print("\nNo data could be processed from any directory.")
        return
    
    # Finalize the plot
    ax.set_ylabel("Error (m)", fontsize=12)
    ax.set_xlabel("Exclusion radius (m)", fontsize=12)
    ax.set_title("Impact of Exclusion Radius on Localization Error", fontsize=16, weight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    save_path = os.path.join(OUTPUT_DIR, 'multi_folder_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    # Example usage for multiple folders:
    
    # Option 1: Using a list of directories (auto-labeled)
    # analyze_and_plot_results(directories=["results_1", "results_2", "results_3"])
    
    # Option 2: Using a dictionary with custom labels
    analyze_and_plot_results(directories={
        "Distorted": "results/distorted",
        "Undistorted": "results/undistorted"
    })
    
    # Option 3: Single directory (backward compatible)
    #analyze_and_plot_results(directory="results")
    analyze_trajectory(GPS_CSV_PATH)