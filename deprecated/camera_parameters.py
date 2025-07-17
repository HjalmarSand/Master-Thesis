import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- 1. Load and Prepare the Data ---
# Load the CSV file
df = pd.read_csv("distortion_parameters.csv")

# We only care about the first 90 degrees for a stable fit, as the math can get unstable at extremes.
# You can experiment with using more of the data if needed.
df = df[df["y_a"] <= 90].copy()

# Extract the relevant columns
# y_a is our angle in degrees, real_height is our distorted radius
theta_deg = df["y_a"].values
r_distorted_measured = df["real_height"].values

# Convert angle from degrees to radians, as required by the formula
theta_rad = np.deg2rad(theta_deg)


# --- 2. Define the Fisheye Model ---
# This is the formula Metashape uses. We need to find f, k1, k2, k3, k4.
def fisheye_model(theta, f, k1, k2, k3, k4):
    """
    Agisoft Metashape's Fisheye distortion model.
    theta must be in radians.
    """
    theta2 = theta**2
    theta4 = theta**4
    theta6 = theta**6
    theta8 = theta**8

    radial_distortion = 1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8

    return f * theta * radial_distortion


# --- 3. Perform the Curve Fitting ---
# Use scipy's curve_fit to find the best parameters (f, k1, k2, k3, k4)
# that make our fisheye_model function match our measured data.
initial_guesses = [
    1000,
    0,
    0,
    0,
    0,
]  # Start with a reasonable guess for f and zero for k's
popt, pcov = curve_fit(
    fisheye_model, theta_rad, r_distorted_measured, p0=initial_guesses
)

# Extract the optimized parameters
f_fit, k1_fit, k2_fit, k3_fit, k4_fit = popt


# --- 4. Print and Visualize the Results ---
print("--- Agisoft Metashape Fisheye Parameters ---")
print(f"Focal Length (f):  {f_fit:.6f}")
print(f"K1:                {k1_fit:.6f}")
print(f"K2:                {k2_fit:.6f}")
print(f"K3:                {k3_fit:.6f}")
print(f"K4:                {k4_fit:.6f}")
print("---------------------------------------------")
print("NOTE: Use these values in the Camera Calibration dialog in Metashape.")
print("Assume Cx and Cy are 0, and P1 and P2 are 0 unless you know otherwise.")

# Plot the original data vs. the fitted model to see how well it worked
r_distorted_predicted = fisheye_model(theta_rad, f_fit, k1_fit, k2_fit, k3_fit, k4_fit)

plt.figure(figsize=(10, 6))
plt.scatter(
    theta_deg, r_distorted_measured, label="Measured Data from CSV", s=10, c="blue"
)
plt.plot(
    theta_deg,
    r_distorted_predicted,
    label="Fitted Fisheye Model",
    color="red",
    linewidth=2,
)
plt.xlabel("Angle from Center (degrees)")
plt.ylabel("Distorted Radius (pixels or mm)")
plt.title("Fisheye Lens Model Fit")
plt.legend()
plt.grid(True)
plt.show()
