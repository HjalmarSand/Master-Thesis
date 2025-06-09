import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import constants
GPS_CSV_PATH = constants.GPS_LOCATION_AND_ORIENTATION  
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

if __name__ == "__main__":
    analyze_trajectory(GPS_CSV_PATH)