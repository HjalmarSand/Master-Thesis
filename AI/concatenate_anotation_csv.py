import os

import numpy as np
import pandas as pd


def process_annotation_file(filepath, dataset_num, index_offset):
    """
    Processes a 'roll_annotations*.csv' file. These files are the primary source of truth.
    """
    if not os.path.exists(filepath):
        print(f"Info: File not found, skipping: {os.path.basename(filepath)}")
        return None

    try:
        df = pd.read_csv(filepath, header=None, skip_blank_lines=True)
        if df.empty:
            return None

        num_cols = len(df.columns)
        col_names = ["raw_id", "roll_back", "roll_front"]
        df.columns = col_names[:num_cols]

        if "roll_front" not in df.columns:
            df["roll_front"] = np.nan

        temp_numeric = pd.to_numeric(df["raw_id"], errors="coerce")
        is_float = (temp_numeric.notna()) & (temp_numeric != temp_numeric.round())
        misplaced_angle_mask = df["roll_back"].isnull() & is_float
        df.loc[misplaced_angle_mask, "roll_back"] = df.loc[
            misplaced_angle_mask, "raw_id"
        ]

        df["dataset"] = dataset_num
        df["index"] = df.index + index_offset

        df["roll_back"] = pd.to_numeric(df["roll_back"], errors="coerce")
        df["roll_front"] = pd.to_numeric(df["roll_front"], errors="coerce")

        # Invert the sign of the back camera roll as per unit circle convention to left tilts
        df["roll_back"] = df["roll_back"] * -1

        print(f"Processed (high priority): {os.path.basename(filepath)}")
        return df[["dataset", "index", "roll_back", "roll_front"]]

    except pd.errors.EmptyDataError:
        print(f"Warning: {os.path.basename(filepath)} is empty.")
        return None


def process_roll_file(filepath, dataset_num):
    """
    Processes a 'roll*.csv' file. These are a secondary source of data.
    """
    if not os.path.exists(filepath):
        print(f"Info: File not found, skipping: {os.path.basename(filepath)}")
        return None

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None

        df.rename(columns={"image": "index", "roll": "roll_back"}, inplace=True)
        df["dataset"] = dataset_num
        df["roll_front"] = np.nan

        df["roll_back"] = pd.to_numeric(df["roll_back"], errors="coerce")

        df["roll_back"] = df["roll_back"] * -1

        print(f"Processed (low priority):  {os.path.basename(filepath)}")
        return df[["dataset", "index", "roll_back", "roll_front"]]

    except pd.errors.EmptyDataError:
        print(f"Warning: {os.path.basename(filepath)} is empty.")
        return None


def create_master_annotations():
    """
    Reads, processes, de-duplicates, and combines annotation files into a master CSV.
    Data from 'roll_annotations*.csv' files is given priority over 'roll*.csv' files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "roll_annotations.csv"
    output_path = os.path.join(script_dir, output_filename)

    # Process files in two batches: high priority and low priority
    priority_dfs = []
    secondary_dfs = []

    priority_dfs.append(
        process_annotation_file(
            os.path.join(script_dir, "roll_annotations1.csv"), 1, 801
        )
    )
    priority_dfs.append(
        process_annotation_file(
            os.path.join(script_dir, "roll_annotations2.csv"), 2, 400
        )
    )
    priority_dfs.append(
        process_annotation_file(
            os.path.join(script_dir, "roll_annotations3.csv"), 3, 500
        )
    )

    secondary_dfs.append(process_roll_file(os.path.join(script_dir, "roll2.csv"), 2))
    secondary_dfs.append(process_roll_file(os.path.join(script_dir, "roll3.csv"), 3))

    # Filter out None values from both lists
    priority_dfs = [df for df in priority_dfs if df is not None]
    secondary_dfs = [df for df in secondary_dfs if df is not None]

    all_dfs = priority_dfs + secondary_dfs

    if not all_dfs:
        print("No data files found or processed. Exiting.")
        return

    try:
        # Combine high-priority dataframes first, then low-priority
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Keep the FIRST occurrence of each (dataset, index) pair.
        # Since priority_dfs were concatenated first, their data will be kept.
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(
            subset=["dataset", "index"], keep="first", inplace=True
        )
        final_rows = len(combined_df)

        if initial_rows > final_rows:
            print(
                f"\nDe-duplication complete: Removed {initial_rows - final_rows} redundant entries."
            )

        # Drop rows where BOTH roll_back and roll_front are missing
        combined_df.dropna(subset=["roll_back", "roll_front"], how="all", inplace=True)

        # Reorder and sort the final dataframe
        final_df = combined_df[["dataset", "index", "roll_back", "roll_front"]]
        final_df = final_df.sort_values(by=["dataset", "index"]).reset_index(drop=True)

        # Ensure 'dataset' and 'index' columns are integers
        final_df["dataset"] = final_df["dataset"].astype(int)
        final_df["index"] = final_df["index"].astype(int)

        final_df.to_csv(output_path, index=False)

        print("\nSuccessfully processed and combined the files.")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during final processing: {e}")


if __name__ == "__main__":
    create_master_annotations()
