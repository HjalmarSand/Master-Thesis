# megaloc_querier.py

import os
import pickle
import re # Import the regular expression module
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cdist

class MegaLocQuerier:
    """
    Class to query a pre-computed MegaLoc database with one or more panoramic images.
    """

    def __init__(self, database_path, gps_csv_path):
        """
        Initialize the querier.

        Args:
            database_path (str): Path to the MegaLoc database file (.pkl).
            gps_csv_path (str): Path to the GPS CSV file for the query images.
        """
        self.database_path = database_path
        self.gps_csv_path = gps_csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.database = None
        self.db_features = None
        self.db_info = None
        self.query_gps_df = None

        print(f"MegaLocQuerier initialized. Using device: {self.device}")

    def initialize_model_and_transforms(self):
        """Initialize the MegaLoc model and image transforms."""
        if self.model is not None:
            return

        print("Initializing MegaLoc model for querying...")
        try:
            self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            self.model.eval()
            self.model.to(self.device)

            self.preprocess = T.Compose([
                T.Resize((480, 640)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("MegaLoc model and transforms initialized.")
        except Exception as e:
            print(f"Error loading MegaLoc model from torch.hub: {e}")
            raise

    def load_database(self):
        """Load the MegaLoc feature database."""
        if self.database is not None:
            return

        print(f"Loading database from {self.database_path}...")
        try:
            with open(self.database_path, 'rb') as f:
                self.database = pickle.load(f)

            self.db_features = np.array([entry['features'] for entry in self.database])
            self.db_info = [{'path': entry['path'], 'easting': entry['easting'], 'northing': entry['northing']} for entry in self.database]
            print(f"Database loaded with {len(self.database)} entries.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Database file not found at {self.database_path}")

    def load_query_gps_data(self):
        """Load GPS data for the query images."""
        if self.query_gps_df is not None:
            return

        try:
            self.query_gps_df = pd.read_csv(self.gps_csv_path)
            # Ensure the index is the 'filename' column, which contains keys like '0400.jpg'
            if 'filename' in self.query_gps_df.columns:
                 self.query_gps_df.set_index("filename", inplace=True)
            else:
                raise KeyError("'filename' column not found in the query GPS CSV.")
            print(f"Loaded query GPS data with {len(self.query_gps_df)} entries.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Query GPS CSV file not found at {self.gps_csv_path}")

    # --- NEW HELPER METHOD ---
    # This method mirrors the logic from the MegaLocDatabaseCreator to ensure
    # consistency between how database keys and query keys are generated.
    def _extract_key_from_filename(self, img_path):
        """
        Extracts the CSV key (e.g., '0400.jpg') from a full image path
        that might have a prefix (e.g., '.../panorama_0400.jpg').
        """
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Use regex to find the last 4-digit number in the filename.
        match = re.search(r"(\d{4})$", name_without_ext)
        if match:
            # Return the matched number with a .jpg extension to match the CSV key.
            return match.group(1) + ".jpg"
        else:
            print(f"Warning: Could not extract 4-digit key from {filename}")
            return None

    def extract_features(self, image_path):
        """Extract MegaLoc features from a single image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_t = self.preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0).to(self.device)

            with torch.no_grad():
                features = self.model(batch_t)

            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not process query image {image_path}: {e}")
            return None

    def query_single_image(self, image_path):
        """
        Find the best match in the database for a single query image.
        """
        # Lazy load components on first query
        if self.model is None: self.initialize_model_and_transforms()
        if self.database is None: self.load_database()
        if self.query_gps_df is None: self.load_query_gps_data()

        query_features = self.extract_features(image_path)
        if query_features is None:
            return None

        distances = cdist(np.expand_dims(query_features, axis=0), self.db_features, metric='cosine').squeeze()
        best_match_idx = np.argmin(distances)
        best_match = self.db_info[best_match_idx]

        # --- CHANGED SECTION ---
        # Use the new helper method to get the correct key for CSV lookup.
        query_key = self._extract_key_from_filename(image_path)
        if not query_key:
            print(f"Skipping query for {os.path.basename(image_path)} as key could not be determined.")
            return None

        try:
            # Look up the ground truth using the extracted key (e.g., '0400.jpg')
            query_gps = self.query_gps_df.loc[query_key]
            gt_easting, gt_northing = query_gps['easting'], query_gps['northing']
        except KeyError:
            # This error now gives much more useful debug info.
            print(f"Warning: No GPS data found for key '{query_key}' (derived from {os.path.basename(image_path)}).")
            return None
        # --- END CHANGED SECTION ---

        match_coords = np.array([best_match['easting'], best_match['northing']])
        query_coords = np.array([gt_easting, gt_northing])
        error_meters = np.linalg.norm(match_coords - query_coords)

        return {
            "query_path": image_path,
            "best_match_path": best_match['path'],
            "localization_error_meters": error_meters,
            "cosine_distance": distances[best_match_idx],
        }

    def query_folder(self, folder_path):
        """
        Query the database with all images in a folder and calculate aggregate stats.
        """
        image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_paths:
            print(f"No images found in {folder_path}.")
            return None

        print(f"Querying with {len(image_paths)} images from {folder_path}...")

        results = []
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Processing query image {i+1}/{len(image_paths)}...")
            result = self.query_single_image(img_path)
            if result:
                results.append(result)

        if not results:
            print("\nError: No successful queries were made. Check file names and GPS data.")
            return None

        errors = [r['localization_error_meters'] for r in results]
        avg_error = np.mean(errors)
        median_error = np.median(errors)
        std_dev_error = np.std(errors)
        
        recall_at_1m = np.mean(np.array(errors) <= 1) * 100
        recall_at_5m = np.mean(np.array(errors) <= 5) * 100
        recall_at_10m = np.mean(np.array(errors) <= 10) * 100

        stats = {
            "total_queries": len(image_paths),
            "successful_queries": len(results),
            "average_error_meters": avg_error,
            "median_error_meters": median_error,
            "std_dev_error_meters": std_dev_error,
            "recall_at_1m_percent": recall_at_1m,
            "recall_at_5m_percent": recall_at_5m,
            "recall_at_10m_percent": recall_at_10m,
            "individual_results": results
        }

        return stats