# create_database.py (with MegaLoc)
import os
import glob
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import pickle
import numpy as np

import constants

# TODO: Fix paths now that we use separate folders for front and back images. Also move paths to constants.py

GPS_CSV_PATH = "Hjalmar/dataset1/gps_interpolated_with_orientation.csv"
DATABASE_OUTPUT_FILE = "image_database_megaloc.pkl"
MEGALOC_INPUT_IMAGE_SIZE = (480, 640) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_model_and_transforms():
    print("Initializing MegaLoc model...")
    try:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    except Exception as e:
        print(f"Error loading MegaLoc model from torch.hub: {e}")
        print("Ensure you have an internet connection and the repository 'gmberton/MegaLoc' is accessible.")
        exit()
    
    model.eval()
    model.to(device)

    # Define the preprocessing transforms
    # Resize to MegaLoc's expected input size, convert to tensor, and normalize
    preprocess = T.Compose([
        T.Resize(MEGALOC_INPUT_IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("MegaLoc model and transforms initialized.")
    return model, preprocess

def extract_features(image_path, model, preprocess_fn):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess_fn(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        with torch.no_grad():
            features = model(batch_t) 

        features_flat = features.squeeze().cpu().numpy() # MegaLoc outputs a 1D feature vector
        return features_flat
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not process image {image_path}: {e}")
        return None

def main():
    try:
        gps_df = pd.read_csv(GPS_CSV_PATH)
        gps_df.set_index('filename', inplace=True)
    except FileNotFoundError:
        print(f"Error: GPS CSV file not found at {GPS_CSV_PATH}"); return
    except KeyError:
        print(f"Error: 'filename' column not found in {GPS_CSV_PATH}."); return

    model, preprocess_fn = initialize_model_and_transforms()
    database_entries = []
    print("Processing database images and extracting MegaLoc features...")

    db_image_paths = glob.glob(os.path.join(constants.CAM1_DISTORTED_FOLDER, "*.jpg"))
    if not db_image_paths:
        print(f"Error: No JPG images found in {constants.CAM1_DISTORTED_FOLDER}"); return

    for i, img_path in enumerate(db_image_paths):
        print(f"Processing image {i+1}/{len(db_image_paths)}: {os.path.basename(img_path)}")
        base_filename_parts = os.path.basename(img_path).split('_')
        original_base_name = base_filename_parts[0] + ".jpg" if len(base_filename_parts) > 0 else os.path.basename(img_path)

        if original_base_name in gps_df.index:
            gps_data = gps_df.loc[original_base_name]
            features = extract_features(img_path, model, preprocess_fn)
            if features is not None:
                database_entries.append({
                    "path": img_path, "base_name": original_base_name,
                    "easting": gps_data["easting"], "northing": gps_data["northing"],
                    "orientation": gps_data["Orientation"], "features": features
                })
        else:
            print(f"Warning: No GPS data for {original_base_name} (from {img_path}). Skipping.")

    if not database_entries:
        print("Error: No features extracted."); return

    try:
        with open(DATABASE_OUTPUT_FILE, 'wb') as f:
            pickle.dump(database_entries, f)
        print(f"Processed {len(database_entries)} images. Database saved to {DATABASE_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving database: {e}")

if __name__ == "__main__":
    main()