import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# --- Configuration ---
INPUT_CSV_PATH = "images/dataset1/gps_interpolated_with_orientation.csv"
OUTPUT_CSV_PATH = "images/dataset1/gps_interpolated_spline_orientation.csv"


def calculate_orientation_diff(df, easting_col="easting", northing_col="northing"):
    """Your original diff-based orientation calculation for comparison."""
    d_easting = df[easting_col].diff().shift(-1)
    d_northing = df[northing_col].diff().shift(-1)
    angle = np.degrees(np.arctan2(d_easting, d_northing))
    orientation_deg = (angle + 360) % 360
    epsilon = 1e-9
    stationary_mask = (np.abs(d_easting) < epsilon) & (np.abs(d_northing) < epsilon)
    orientation_deg[stationary_mask] = np.nan
    return orientation_deg.ffill().bfill().rename("Orientation_Diff")


def calculate_orientation_spline(
    df, n_col="n", easting_col="easting", northing_col="northing"
):
    """
    Calculates orientation using cubic spline interpolation for a smooth,
    accurate, and instantaneous heading.

    Returns:
        pd.Series: A Pandas Series containing the orientation in degrees.
    """
    print("Calculating orientation using cubic splines...")

    # Splines require a strictly increasing 'x' axis (our time/n column) and no NaNs.
    # We'll drop any duplicate time steps and interpolate missing GPS values.
    df_clean = df.drop_duplicates(subset=n_col).copy()
    df_clean[easting_col] = df_clean[easting_col].interpolate(method="linear")
    df_clean[northing_col] = df_clean[northing_col].interpolate(method="linear")

    if len(df_clean) < 4:
        print(
            "Error: Not enough data points (<4) to fit a cubic spline. Falling back to diff method."
        )
        return calculate_orientation_diff(df, easting_col, northing_col)

    # Our time axis
    t = df_clean[n_col]

    # Fit cubic splines to the easting and northing data
    spline_easting = CubicSpline(t, df_clean[easting_col])
    spline_northing = CubicSpline(t, df_clean[northing_col])

    # Get the first derivative of the splines (d(pos)/dt = velocity)
    spline_e_deriv = spline_easting.derivative(nu=1)
    spline_n_deriv = spline_northing.derivative(nu=1)

    # Evaluate the derivatives at our time points to get the velocity vectors
    d_easting = spline_e_deriv(t)
    d_northing = spline_n_deriv(t)

    # Calculate the angle of the velocity vector
    angle_rad = np.arctan2(d_easting, d_northing)
    orientation_deg = (np.degrees(angle_rad) + 360) % 360

    # Reindex to match the original dataframe in case we dropped rows
    return (
        pd.Series(orientation_deg, index=df_clean.index)
        .reindex(df.index)
        .ffill()
        .bfill()
        .rename("Orientation")
    )


# --- Main Execution ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV_PATH} not found.")
        exit()

    # Calculate orientation using both methods
    df["Orientation_Spline"] = calculate_orientation_spline(df)
    df["Orientation_Diff"] = calculate_orientation_diff(df)

    # The 'Orientation' column should be our best estimate
    df["Orientation"] = df["Orientation_Spline"]

    # Save the new, improved dataframe
    # We only need the final, best 'Orientation' column for the next step.
    output_df = df[["n", "filename", "easting", "northing", "Orientation"]].copy()
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(
        f"\nSuccessfully saved new data with spline-based orientation to:\n{OUTPUT_CSV_PATH}"
    )

    # --- Plotting for visual comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Trajectory
    axes[0].plot(
        df["easting"], df["northing"], marker=".", linestyle="-", label="Trajectory"
    )
    axes[0].set_title("Boat Trajectory")
    axes[0].set_xlabel("Easting")
    axes[0].set_ylabel("Northing")
    axes[0].grid(True, linestyle="--")
    axes[0].axis("equal")
    axes[0].legend()

    # Plot 2: Orientation Comparison
    axes[1].plot(
        df["n"],
        df["Orientation_Diff"],
        "r.-",
        label="Orientation (Diff Method)",
        alpha=0.7,
    )
    axes[1].plot(
        df["n"],
        df["Orientation_Spline"],
        "b-",
        label="Orientation (Spline Method)",
        linewidth=2,
    )
    axes[1].set_title("Orientation Calculation Comparison")
    axes[1].set_xlabel("Time Step (n)")
    axes[1].set_ylabel("Orientation (Degrees from North)")
    axes[1].grid(True, linestyle="--")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
