import os

import pandas as pd


def process_csv_files():
    """
    Reads four specific CSV files from the script's directory, processes them,
    and combines them into a single CSV file named 'roll_anotation.csv'.

    - 'roll_anotation1.csv' & 'roll_anotation2.csv': No headers, processed for datasets 1 & 2.
    - 'roll2.csv' & 'roll3.csv': Have headers ('image', 'roll'), processed for datasets 2 & 3.

    The script handles malformed rows and filters out entries missing an angle.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "roll_annotations.csv"
    output_path = os.path.join(script_dir, output_filename)

    # Define all source file paths
    files_to_process = {
        "anot1": os.path.join(script_dir, "roll_annotations1.csv"),
        "anot2": os.path.join(script_dir, "roll_annotations2.csv"),
        "roll2": os.path.join(script_dir, "roll2.csv"),
        "roll3": os.path.join(script_dir, "roll3.csv"),
    }

    all_dfs = []

    #  2. Process Each File
    try:
        #  Process roll_anotation1.csv (Dataset 1)
        if os.path.exists(files_to_process["anot1"]):
            df1 = pd.read_csv(
                files_to_process["anot1"],
                header=None,
                names=["dataset", "angle"],
                skip_blank_lines=True,
            )
            temp_numeric_1 = pd.to_numeric(df1["dataset"], errors="coerce")
            is_not_whole_number_1 = (temp_numeric_1.notna()) & (
                temp_numeric_1 != temp_numeric_1.round()
            )
            misplaced_angle_mask_1 = df1["angle"].isnull() & is_not_whole_number_1
            df1.loc[misplaced_angle_mask_1, "angle"] = df1.loc[
                misplaced_angle_mask_1, "dataset"
            ]
            df1["dataset"] = 1
            df1["index"] = df1.index + 801
            all_dfs.append(df1)
            print("Processed roll_anotation1.csv")

        #  Process roll_anotation2.csv (Dataset 2)
        if os.path.exists(files_to_process["anot2"]):
            df2 = pd.read_csv(
                files_to_process["anot2"],
                header=None,
                names=["dataset", "angle"],
                skip_blank_lines=True,
            )
            temp_numeric_2 = pd.to_numeric(df2["dataset"], errors="coerce")
            is_not_whole_number_2 = (temp_numeric_2.notna()) & (
                temp_numeric_2 != temp_numeric_2.round()
            )
            misplaced_angle_mask_2 = df2["angle"].isnull() & is_not_whole_number_2
            df2.loc[misplaced_angle_mask_2, "angle"] = df2.loc[
                misplaced_angle_mask_2, "dataset"
            ]
            df2["dataset"] = 2
            df2["index"] = df2.index + 400
            all_dfs.append(df2)
            print("Processed roll_anotation2.csv")

        #  Process roll2.csv (Dataset 2)
        if os.path.exists(files_to_process["roll2"]):
            df_roll2 = pd.read_csv(files_to_process["roll2"])
            df_roll2.rename(columns={"image": "index", "roll": "angle"}, inplace=True)
            df_roll2["dataset"] = 2
            all_dfs.append(df_roll2)
            print("Processed roll2.csv")

        #  Process roll3.csv (Dataset 3)
        if os.path.exists(files_to_process["roll3"]):
            df_roll3 = pd.read_csv(files_to_process["roll3"])
            df_roll3.rename(columns={"image": "index", "roll": "angle"}, inplace=True)
            df_roll3["dataset"] = 3
            all_dfs.append(df_roll3)
            print("Processed roll3.csv")

        if not all_dfs:
            print("No data files found to process. Exiting.")
            return

        #  3. Combine and Save
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Filter out rows with no angle value
        combined_df.dropna(subset=["angle"], inplace=True)

        # Reorder and sort the final dataframe
        final_df = combined_df[["dataset", "index", "angle"]]
        final_df = final_df.sort_values(by="index").reset_index(drop=True)

        # Ensure 'dataset' and 'index' columns are integers
        final_df["dataset"] = final_df["dataset"].astype(int)
        final_df["index"] = final_df["index"].astype(int)

        final_df.to_csv(output_path, index=False)

        print("\nSuccessfully processed and combined the files.")
        print(f"Output saved to: {output_path}")

    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the CSV files is empty or improperly formatted. {e}")
    except Exception as e:
        print(f"An error occurred during file processing: {e}")


if __name__ == "__main__":
    process_csv_files()
