# skyline_database.py
import json
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple

class SkylineDatabase:
    """
    Manages a database of skyline vectors associated with image identifiers.
    """
    def __init__(self, db_file_path: Optional[str] = None):
        """
        Initializes the SkylineDatabase.

        Args:
            db_file_path (Optional[str]): Path to a file to load/save the database.
        """
        self.db_file_path = db_file_path
        self.skylines: Dict[str, Dict[str, Any]] = {} # image_id -> {"vector": np.array, "metadata": {}}
        if db_file_path and os.path.exists(db_file_path):
            self.load_database(db_file_path)

    def add_skyline(self, image_id: str, skyline_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a skyline vector and its metadata to the database.

        Args:
            image_id (str): A unique identifier for the image (e.g., filename or UUID).
            skyline_vector (np.ndarray): The 1D skyline vector.
            metadata (Optional[Dict[str, Any]]): Additional information (e.g., original image path, GPS).
        """
        if not isinstance(skyline_vector, np.ndarray):
            raise ValueError("skyline_vector must be a NumPy array.")
        
        self.skylines[image_id] = {
            "vector": skyline_vector,
            "metadata": metadata if metadata is not None else {}
        }
        print(f"Added/Updated skyline for image_id: {image_id}")

    def get_skyline_data(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the skyline vector and metadata for a given image ID.

        Args:
            image_id (str): The identifier of the image.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing 'vector' and 'metadata', or None if not found.
        """
        return self.skylines.get(image_id)

    def get_all_skylines_with_ids(self) -> List[Tuple[str, np.ndarray]]:
        """
        Returns a list of all (image_id, skyline_vector) pairs in the database.
        """
        return [(img_id, data["vector"]) for img_id, data in self.skylines.items()]

    def count(self) -> int:
        """Returns the number of skylines in the database."""
        return len(self.skylines)

    def save_database(self, file_path: Optional[str] = None):
        """
        Saves the current database to a file (e.g., JSON for portability, with numpy arrays as lists).
        For larger databases, consider formats like HDF5 or dedicated DBs.

        Args:
            file_path (Optional[str]): Path to save the database. Uses instance's db_file_path if None.
        """
        path_to_save = file_path or self.db_file_path
        if not path_to_save:
            print("Error: No file path provided for saving the database.")
            return

        db_to_save = {}
        for img_id, data in self.skylines.items():
            db_to_save[img_id] = {
                "vector": data["vector"].tolist(), # Convert numpy array to list for JSON compatibility
                "metadata": data["metadata"]
            }
        
        try:
            with open(path_to_save, 'w') as f:
                json.dump(db_to_save, f, indent=4)
            print(f"Database saved to {path_to_save}")
        except Exception as e:
            print(f"Error saving database to {path_to_save}: {e}")

    def load_database(self, file_path: Optional[str] = None):
        """
        Loads the database from a file.

        Args:
            file_path (Optional[str]): Path to load the database from. Uses instance's db_file_path if None.
        """
        path_to_load = file_path or self.db_file_path
        if not path_to_load:
            print("Error: No file path provided for loading the database.")
            return
        
        if not os.path.exists(path_to_load):
            print(f"Database file not found at {path_to_load}. Starting with an empty database.")
            self.skylines = {}
            return

        try:
            with open(path_to_load, 'r') as f:
                loaded_db = json.load(f)
            
            self.skylines = {}
            for img_id, data in loaded_db.items():
                self.skylines[img_id] = {
                    "vector": np.array(data["vector"], dtype=np.int32), # Convert list back to numpy array
                    "metadata": data["metadata"]
                }
            print(f"Database loaded from {path_to_load}. Contains {self.count()} entries.")
        except Exception as e:
            print(f"Error loading database from {path_to_load}: {e}. Starting with an empty database.")
            self.skylines = {}