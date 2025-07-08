# create_database.py
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import constants
from SAM_utils import setup_sam_predictor, extract_skyline_from_image

def main():
    parser = argparse.ArgumentParser(description="Build a skyline database from a folder of images.")
    parser.add_argument("--image_dir", required=True, help="Path to the directory with reference images.")
    parser.add_argument("--metadata_csv", default=constants.GPS_LOCATION_AND_ORIENTATION, help="Path to the CSV file with GPS metadata.")
    parser.add_argument("--output_file", default="skyline_database.json", help="Path to save the output JSON database.")
    parser.add_argument("--sam_checkpoint", default="sam_vit_b_01ec64.pth", help="Path to the SAM model checkpoint.")
    args = parser.parse_args()

    # 1. Setup
    predictor = setup_sam_predictor(checkpoint_path=args.sam_checkpoint)
    metadata_df = pd.read_csv(args.metadata_csv).set_index('filename')
    database_entries = []

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(image_files)} images in {args.image_dir}. Processing...")

    # 2. Loop through images and build database
    for filename in tqdm(image_files, desc="Building Database"):
        if filename not in metadata_df.index:
            print(f"Warning: No metadata found for {filename}. Skipping.")
            continue

        image_path = os.path.join(args.image_dir, filename)
        skyline_vector = extract_skyline_from_image(image_path, predictor)

        if skyline_vector is not None:
            location_data = metadata_df.loc[filename]
            entry = {
                "image_path": image_path,
                "gps": {
                    "northing": location_data['northing'],
                    "easting": location_data['easting']
                },
                "skyline_vector": skyline_vector.tolist() # Convert to list for JSON
            }
            database_entries.append(entry)

    # 3. Save the database to a file
    with open(args.output_file, 'w') as f:
        json.dump(database_entries, f, indent=2)
    
    print(f"\nDatabase creation complete. {len(database_entries)} entries saved to {args.output_file}")

if __name__ == "__main__":
    main()