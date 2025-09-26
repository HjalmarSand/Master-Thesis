import numpy as np
import pandas as pd

# Short script to get some data of the error in the roll annotations first we get error before correction

df = pd.read_csv("AI/roll_annotations.csv")

df["roll_back"] = pd.to_numeric(df["roll_back"], errors="coerce")
df["roll_front"] = pd.to_numeric(df["roll_front"], errors="coerce")

# We want the magnitude so take abs of all values
abs_back = df["roll_back"].abs()
abs_front = df["roll_front"].abs()

avg_abs_back = abs_back.mean()
avg_abs_front = abs_front.mean()

avg_abs_sum = (avg_abs_back + avg_abs_front) / 2

print(f"Average |roll_back|: {avg_abs_back}")
print(f"Average |roll_front|: {avg_abs_front}")
print(f"Average (|roll_back|+|roll_front|)/2: {avg_abs_sum}\n")

grouped = df.groupby("dataset").apply(
    lambda g: pd.Series(
        {
            "avg_abs_back": g["roll_back"].abs().mean(),
            "avg_abs_front": g["roll_front"].abs().mean(),
            "avg_abs_sum": (g["roll_back"].abs().mean() + g["roll_front"].abs().mean())
            / 2,
        }
    )
)

print("Per-dataset averages (absolute):")
print(grouped)

# Get the errors after the roll correction
errors_df = pd.read_csv("AI/evaluation_results/evaluation_error_report.csv")

errors_df["dataset"] = errors_df["image_path"].str.extract(r"(dataset\d+)")

errors_df["absolute_error"] = pd.to_numeric(
    errors_df["absolute_error"], errors="coerce"
)

overall_mean_error = errors_df["absolute_error"].mean()

per_dataset_error = errors_df.groupby("dataset")["absolute_error"].mean()

print("\nOverall mean absolute error across all datasets:")
print(overall_mean_error)

print("\nMean absolute error per dataset:")
print(per_dataset_error)
