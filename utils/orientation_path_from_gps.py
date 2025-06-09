import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os 
import constants

def calculate_orientation(df, easting_col='easting', northing_col='northing'):
    """
    Calculates the orientation (heading) of a boat from easting and northing coordinates.

    Args:
        df (pd.DataFrame): DataFrame with easting and northing columns.
        easting_col (str): Name of the easting column.
        northing_col (str): Name of the northing column.

    Returns:
        pd.Series: A Pandas Series containing the orientation in degrees (0-360, clockwise from North).
    """

    # Calculate the difference in easting and northing between consecutive points
    d_easting = df[easting_col].diff().shift(-1) 
    d_northing = df[northing_col].diff().shift(-1)

    # Calculate the angle in degrees
    angle = np.degrees(np.arctan2(d_easting, d_northing))
    orientation_deg = (angle + 360) % 360

    # Handle stationary points: if d_easting and d_northing are both zero for a segment,
    epsilon = 1e-9
    stationary_mask = (np.abs(d_easting) < epsilon) & (np.abs(d_northing) < epsilon)
    orientation_deg[stationary_mask] = np.nan

    # The last point has no 'next' point, so d_easting/d_northing (and thus orientation_deg) will be NaN.
    orientation_deg_filled = orientation_deg.ffill()

    # If the very first point(s) resulted in NaN (e.g., initial stationary period)
    if orientation_deg_filled.isnull().all() and len(df) > 1:
        print("Warning: Could not determine orientation for any segment. Points might be (nearly) identical or only one distinct coordinate pair.")

    return orientation_deg_filled.rename('Orientation')


def plot_trajectory(df, easting_col='easting', northing_col='northing', title_suffix=""):
    """
    Plots the boat's trajectory.

    Args:
        df (pd.DataFrame): DataFrame with easting and northing columns.
        easting_col (str): Name of the easting column.
        northing_col (str): Name of the northing column.
        title_suffix (str): Suffix to add to the plot title.
    """
    if df.empty or easting_col not in df.columns or northing_col not in df.columns:
        print(f"Cannot plot trajectory. DataFrame is empty or missing '{easting_col}'/'{northing_col}' columns.")
        return
    if df[easting_col].isnull().all() or df[northing_col].isnull().all():
        print(f"Cannot plot trajectory. '{easting_col}' or '{northing_col}' column contains all NaN values.")
        return

    plt.figure(figsize=(12, 10)) 
    plt.plot(df[easting_col], df[northing_col], marker='.', markersize=3, linestyle='-', color='blue', label='Trajectory Path', alpha=0.7)
    plt.xlabel(f"{easting_col} (units)")
    plt.ylabel(f"{northing_col} (units)")
    plt.title(f'Boat Trajectory {title_suffix}'.strip())
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend()
    plt.axis('equal')
    plt.tight_layout() 
    plt.show()


if __name__ == "__main__":
    BASE_PATH = "." 
    RELATIVE_PATH_TO_CSV = constants.GPS_LOCATION_AND_ORIENTATION
    INPUT_CSV_FILE = os.path.join(BASE_PATH, RELATIVE_PATH_TO_CSV)
    input_dir = os.path.dirname(INPUT_CSV_FILE)
    input_filename_base = os.path.splitext(os.path.basename(INPUT_CSV_FILE))[0]
    OUTPUT_CSV_FILE = os.path.join(input_dir, f"{input_filename_base}_with_orientation.csv")

    easting_COLUMN_NAME = 'easting'  
    northing_COLUMN_NAME = 'northing'

    print(f"Attempting to load data from: {INPUT_CSV_FILE}")

    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at '{INPUT_CSV_FILE}'.")
        exit()
    except Exception as e:
        print(f"Error reading CSV file '{INPUT_CSV_FILE}': {e}")
        exit()

    missing_cols = []
    if easting_COLUMN_NAME not in df.columns:
        missing_cols.append(easting_COLUMN_NAME)
    if northing_COLUMN_NAME not in df.columns:
        missing_cols.append(northing_COLUMN_NAME)

    if missing_cols:
        print(f"\nError: The CSV file is missing the required coordinate column(s): {', '.join(missing_cols)}")
        exit()

    # Check if coercion resulted in NaNs, which might indicate bad data
    if df[easting_COLUMN_NAME].isnull().any() or df[northing_COLUMN_NAME].isnull().any():
        print("Warning: Some easting/northing values were non-numeric and have been converted to NaN.")
        print(f"Number of NaNs in '{easting_COLUMN_NAME}': {df[easting_COLUMN_NAME].isnull().sum()}")
        print(f"Number of NaNs in '{northing_COLUMN_NAME}': {df[northing_COLUMN_NAME].isnull().sum()}")


    print("\nCalculating orientation...")
    df['Orientation'] = calculate_orientation(df,
                                            easting_col=easting_COLUMN_NAME,
                                            northing_col=northing_COLUMN_NAME)

    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.3f') # Save floats with 3 decimal places
        print(f"\nDataFrame with orientation saved to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Error saving data to '{OUTPUT_CSV_FILE}': {e}")

    plot_trajectory(df,
                    easting_col=easting_COLUMN_NAME,
                    northing_col=northing_COLUMN_NAME,
                    title_suffix=f"from {os.path.basename(INPUT_CSV_FILE)}")