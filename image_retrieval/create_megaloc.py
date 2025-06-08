# create_megaloc.py - Class-based database creation with MegaLoc
import os
import glob
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import pickle
import numpy as np
import constants

class MegaLocDatabaseCreator:
    """Class for creating and managing MegaLoc feature databases."""
    
    def __init__(self, 
                gps_csv_path=None,
                database_output_file="megaloc_db.pkl",
                input_image_size=(480, 640)):
        """
        Initialize the database creator.
        
        Args:
            gps_csv_path: Path to GPS CSV file (defaults to constants.DATASET1_CSV)
            database_output_file: Output database filename
            input_image_size: MegaLoc input image size
        """
        self.gps_csv_path = gps_csv_path or constants.DATASET1_CSV
        self.database_output_file = database_output_file
        self.input_image_size = input_image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.gps_df = None
        
        print(f"MegaLocDatabaseCreator initialized. Using device: {self.device}")
    
    def initialize_model_and_transforms(self):
        """Initialize MegaLoc model and transforms."""
        if self.model is not None:
            return  # Already initialized
            
        print("Initializing MegaLoc model...")
        try:
            self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
        except Exception as e:
            print(f"Error loading MegaLoc model from torch.hub: {e}")
            print("Ensure you have an internet connection and the repository 'gmberton/MegaLoc' is accessible.")
            raise
        
        self.model.eval()
        self.model.to(self.device)
        
        self.preprocess = T.Compose([
            T.Resize(self.input_image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("MegaLoc model and transforms initialized.")
    
    def load_gps_data(self):
        """Load GPS data from CSV file."""
        if self.gps_df is not None:
            return  # Already loaded
            
        try:
            self.gps_df = pd.read_csv(self.gps_csv_path)
            self.gps_df.set_index('filename', inplace=True)
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
    
    def process_camera_images(self, camera_name, camera_path):
        """Process all images from a specific camera and return database entries."""
        database_entries = []
        
        db_image_paths = glob.glob(os.path.join(camera_path, "*.jpg"))
        if not db_image_paths:
            print(f"Warning: No JPG images found in {camera_path} for {camera_name}")
            return database_entries
        
        print(f"\nProcessing {camera_name} images from {camera_path}")
        print(f"Found {len(db_image_paths)} images to process")
        
        for i, img_path in enumerate(db_image_paths):
            if (i + 1) % 10 == 0 or i == 0:  # Progress indicator every 10 images
                print(f"Processing {camera_name} image {i+1}/{len(db_image_paths)}")
            
            base_filename_parts = os.path.basename(img_path).split('_')
            original_base_name = base_filename_parts[1] if len(base_filename_parts) > 0 else os.path.basename(img_path)
            
            if original_base_name in self.gps_df.index:
                gps_data = self.gps_df.loc[original_base_name]
                features = self.extract_features(img_path)
                if features is not None:
                    database_entries.append({
                        "path": img_path,
                        "base_name": original_base_name,
                        "camera": camera_name,
                        "easting": gps_data["easting"],
                        "northing": gps_data["northing"],
                        "orientation": gps_data["Orientation"],
                        "features": features
                    })
            else:
                print(f"Warning: No GPS data for {original_base_name} (from {img_path}). Skipping.")
        
        return database_entries
    
    def create_database(self, cameras=['CAM1', 'CAM2'], camera_paths=None, force_recreate=False):
        """
        Create the MegaLoc database from specified cameras.
        
        Args:
            cameras: List of camera names to process
            camera_paths: Dictionary mapping camera names to paths (uses constants if None)
            force_recreate: If True, recreate even if database exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if database already exists
        if not force_recreate and os.path.exists(self.database_output_file):
            print(f"Database already exists at {self.database_output_file}. Use force_recreate=True to overwrite.")
            return True
        
        # Set default camera paths if not provided
        if camera_paths is None:
            camera_paths = {
                'CAM1': constants.CAM1_DISTORTED_FOLDER,
                'CAM2': constants.CAM2_DISTORTED_FOLDER
            }
        
        try:
            # Initialize components
            self.load_gps_data()
            self.initialize_model_and_transforms()
            
            all_database_entries = []
            print("Processing database images and extracting MegaLoc features...")
            
            # Process each camera
            for camera in cameras:
                if camera not in camera_paths:
                    print(f"Warning: No path defined for {camera}, skipping.")
                    continue
                    
                entries = self.process_camera_images(camera, camera_paths[camera])
                all_database_entries.extend(entries)
                print(f"Processed {len(entries)} images from {camera}")
            
            if not all_database_entries:
                print("Error: No features extracted from any camera.")
                return False
            
            self.save_database(all_database_entries)
            
            print(f"\nDatabase creation complete!")
            print(f"Total images processed: {len(all_database_entries)}")
            for camera in cameras:
                count = sum(1 for entry in all_database_entries if entry['camera'] == camera)
                print(f"  - {camera}: {count} images")
            
            return True
            
        except Exception as e:
            print(f"Error during database creation: {e}")
            return False
    
    def save_database(self, database_entries):
        """Save database entries to pickle file."""
        try:
            with open(self.database_output_file, 'wb') as f:
                pickle.dump(database_entries, f)
            print(f"Database saved to {self.database_output_file}")
        except Exception as e:
            raise Exception(f"Error saving database: {e}")
    
    def load_database(self):
        """Load existing database from file."""
        try:
            with open(self.database_output_file, 'rb') as f:
                database_entries = pickle.load(f)
            return database_entries
        except FileNotFoundError:
            print(f"Database file not found at {self.database_output_file}")
            return None
        except Exception as e:
            print(f"Error loading database: {e}")
            return None
    
    def database_exists(self):
        """Check if database file exists."""
        return os.path.exists(self.database_output_file)
    
    def get_database_info(self):
        """Get information about the existing database."""
        if not self.database_exists():
            return None
            
        entries = self.load_database()
        if entries is None:
            return None
            
        info = {
            'total_entries': len(entries),
            'cameras': {},
            'file_path': self.database_output_file
        }
        
        for entry in entries:
            camera = entry.get('camera', 'UNKNOWN')
            if camera not in info['cameras']:
                info['cameras'][camera] = 0
            info['cameras'][camera] += 1
        
        return info


def main():
    """Create database using default settings."""
    creator = MegaLocDatabaseCreator()
    success = creator.create_database(cameras=['CAM1', 'CAM2'])
    
    if success:
        info = creator.get_database_info()
        if info:
            print(f"\nDatabase info:")
            print(f"  Total entries: {info['total_entries']}")
            print(f"  Cameras: {info['cameras']}")
    
    return success


if __name__ == "__main__":
    main()