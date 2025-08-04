import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image


class MegaLocDatabaseCreator:
    """
    Class for creating and managing MegaLoc feature databases from a single
    directory of panoramic images.
    """

    def __init__(
        self,
        gps_csv_path=None,
        database_output_file="megaloc_pano_db.pkl",
        input_image_size=(
            480,
            640,
        ),  # All images will be resized to these as megaloc required it
    ):
        """
        Initialize the database creator.

        Args:
            gps_csv_path: Path to GPS CSV file .
            database_output_file: Output database filename for panoramic images.
            input_image_size: MegaLoc input image size.
        """
        self.gps_csv_path = gps_csv_path
        self.database_output_file = database_output_file
        self.input_image_size = input_image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.gps_df = None

        print(
            f"MegaLocDatabaseCreator initialized for panoramic images. Using device: {self.device}"
        )

    def initialize_model_and_transforms(self):
        """Initialize MegaLoc model and transforms."""
        if self.model is not None:
            return  # Already initialized

        print("Initializing MegaLoc model...")
        try:
            self.model = torch.hub.load(
                "gmberton/MegaLoc", "get_trained_model", trust_repo=True
            )
        except Exception as e:
            print(f"Error loading MegaLoc model from torch.hub: {e}")
            print(
                "Ensure you have an internet connection and the repository 'gmberton/MegaLoc' is accessible."
            )
            raise

        self.model.eval()
        self.model.to(self.device)

        self.preprocess = T.Compose(
            [
                T.Resize(self.input_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("MegaLoc model and transforms initialized.")

    def load_gps_data(self):
        """Load GPS data from CSV file."""
        if self.gps_df is not None:
            return  # Already loaded

        try:
            self.gps_df = pd.read_csv(self.gps_csv_path)
            self.gps_df.set_index("filename", inplace=True)
            print(f"Loaded GPS data with {len(self.gps_df)} entries")
        except FileNotFoundError:
            raise FileNotFoundError(f"GPS CSV file not found at {self.gps_csv_path}")
        except KeyError:
            raise KeyError(f"'filename' column not found in {self.gps_csv_path}")

    def extract_features(self, image_path):
        """Extract MegaLoc features from a single image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_t = self.preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0).to(self.device)

            with torch.no_grad():
                features = self.model(batch_t)

            features_flat = features.squeeze().cpu().numpy()
            return features_flat
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            return None
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            return None

    def extract_base_filename(self, img_path):
        """
        Extract the base filename from image path using the last 4 digits pattern.
        Always returns with .jpg extension to match CSV file format.

        Args:
            img_path: Full path to the image file.

        Returns:
            str: Base filename with 4-digit suffix and .jpg extension (e.g., "0801.jpg").
        """
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        match = re.search(r"(\d{4})$", name_without_ext)
        if match:
            return match.group(1) + ".jpg"
        else:
            print(f"Warning: Could not extract 4-digit pattern from {filename}")
            return None

    def create_database(self, image_directory, force_recreate=False):
        """
        Create the MegaLoc database from a single directory of images.

        Args:
            image_directory (str): Path to the directory containing panoramic images.
            force_recreate (bool): If True, recreate even if database exists.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not force_recreate and os.path.exists(self.database_output_file):
            print(
                f"Database already exists at {self.database_output_file}. Use force_recreate=True to overwrite."
            )
            return True

        try:
            # Initialize components
            self.load_gps_data()
            self.initialize_model_and_transforms()

            # Find all images in the directory
            jpg_paths = glob.glob(os.path.join(image_directory, "*.jpg"))
            png_paths = glob.glob(os.path.join(image_directory, "*.png"))
            db_image_paths = sorted(jpg_paths + png_paths)  # Sort for consistent order

            if not db_image_paths:
                print(f"Error: No JPG or PNG images found in {image_directory}")
                return False

            print(
                f"\nProcessing {len(db_image_paths)} images from {image_directory}..."
            )

            all_database_entries = []
            for i, img_path in enumerate(db_image_paths):
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"Processing image {i + 1}/{len(db_image_paths)}")

                base_name = self.extract_base_filename(img_path)

                if base_name and base_name in self.gps_df.index:
                    gps_data = self.gps_df.loc[base_name]
                    features = self.extract_features(img_path)

                    if features is not None:
                        all_database_entries.append(
                            {
                                "path": img_path,
                                "base_name": base_name,
                                "easting": gps_data["easting"],
                                "northing": gps_data["northing"],
                                "features": features,
                            }
                        )
                else:
                    if base_name:
                        print(
                            f"Warning: No GPS data for {base_name} (from {img_path}). Skipping."
                        )
                    else:
                        print(
                            f"Warning: Could not extract valid base name from {img_path}. Skipping."
                        )

            if not all_database_entries:
                print(
                    "Error: No features were extracted. Check image paths and GPS data."
                )
                return False

            self.save_database(all_database_entries)

            print("\nDatabase creation complete!")
            print(
                f"Total images processed and added to database: {len(all_database_entries)}"
            )

            return True

        except Exception as e:
            print(f"An error occurred during database creation: {e}")
            return False

    def save_database(self, database_entries):
        """Save database entries to a pickle file."""
        try:
            with open(self.database_output_file, "wb") as f:
                pickle.dump(database_entries, f)
            print(f"Database saved to {self.database_output_file}")
        except Exception as e:
            raise IOError(f"Error saving database: {e}")

    def load_database(self):
        """Load an existing database from a file."""
        try:
            with open(self.database_output_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Database file not found at {self.database_output_file}")
            return None
        except Exception as e:
            print(f"Error loading database: {e}")
            return None

    def database_exists(self):
        """Check if the database file exists."""
        return os.path.exists(self.database_output_file)

    def get_database_info(self):
        """Get information about the existing database."""
        if not self.database_exists():
            print("Database does not exist.")
            return None

        entries = self.load_database()
        if entries is None:
            return None

        info = {
            "total_entries": len(entries),
            "file_path": self.database_output_file,
            "file_types": {"jpg": 0, "png": 0, "other": 0},
        }

        for entry in entries:
            path = entry.get("path", "")
            if path.lower().endswith(".jpg"):
                info["file_types"]["jpg"] += 1
            elif path.lower().endswith(".png"):
                info["file_types"]["png"] += 1
            else:
                info["file_types"]["other"] += 1

        return info


def main():
    """Create a database using the new panoramic image setup."""

    panorama_dir = "images/dataset1/panoramas"

    creator = MegaLocDatabaseCreator(database_output_file="megaloc_pano_db.pkl")
    success = creator.create_database(image_directory=panorama_dir, force_recreate=True)

    if success:
        info = creator.get_database_info()
        if info:
            print(f"\nDatabase Info:")
            print(f"  File Path: {info['file_path']}")
            print(f"  Total Entries: {info['total_entries']}")
            print(f"  File Types: {info['file_types']}")

    return success


if __name__ == "__main__":
    main()
