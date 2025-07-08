import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def calculate_orientation(df, easting_col='easting', northing_col='northing'):
    """
    Calculates the orientation (heading) of a vehicle from easting and northing coordinates.

    Args:
        df (pd.DataFrame): DataFrame with easting and northing columns.
        easting_col (str): Name of the easting column.
        northing_col (str): Name of the northing column.

    Returns:
        pd.Series: A Pandas Series containing the orientation in degrees (0-360, clockwise from North).
    """
    d_easting = df[easting_col].diff().shift(-1) 
    d_northing = df[northing_col].diff().shift(-1)

    angle = np.degrees(np.arctan2(d_easting, d_northing))
    orientation_deg = (angle + 360) % 360

    epsilon = 1e-9
    stationary_mask = (np.abs(d_easting) < epsilon) & (np.abs(d_northing) < epsilon)
    orientation_deg[stationary_mask] = np.nan

    orientation_deg_filled = orientation_deg.ffill().bfill()

    if orientation_deg_filled.isnull().all() and len(df) > 1:
        print("Warning: Could not determine orientation for any segment.")

    return orientation_deg_filled.rename('Orientation')


def plot_trajectory(df, easting_col='easting', northing_col='northing', title_suffix="", save_path=None):
    """
    Plots the vehicle's trajectory and optionally saves it to a file.

    Args:
        df (pd.DataFrame): DataFrame with easting and northing columns.
        easting_col (str): Name of the easting column.
        northing_col (str): Name of the northing column.
        title_suffix (str): Suffix to add to the plot title.
        save_path (Path or str, optional): Path to save the plot. If None, shows the plot.
    """
    if df.empty or easting_col not in df.columns or northing_col not in df.columns:
        print(f"Cannot plot trajectory. DataFrame is empty or missing columns.")
        return
    if df[easting_col].isnull().all() or df[northing_col].isnull().all():
        print(f"Cannot plot trajectory. Coordinate columns contain all NaN values.")
        return

    plt.figure(figsize=(12, 10)) 
    plt.plot(df[easting_col], df[northing_col], marker='.', markersize=3, linestyle='', color='blue', label='Trajectory Path', alpha=0.7)
    plt.xlabel(f"{easting_col} (meters)")
    plt.ylabel(f"{northing_col} (meters)")
    plt.title(f'Vehicle Trajectory {title_suffix}'.strip())
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend()
    plt.axis('equal')
    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to: {save_path}")
        plt.close() # Close the plot to prevent it from displaying in the console
    else:
        plt.show()