# Image retrieval using colmap that internally uses sift then ann matcher, then geometric verification to find top matches.
# See publication below detailing the methods. This file uses the naive matcher so far, and is infeasible for large datasets,
# but can be used as a baseline that we later compare with speeding up lookup with BoW, probably with MegaLoc that seems promising
# in its description. 
# https://www.researchgate.net/publication/301197096_Structure-from-Motion_Revisited

import subprocess
from pathlib import Path
import sqlite3
import shutil
from typing import List, Tuple, Dict, Optional

import constants

COLMAP_EXE_PATH = "colmap"
COLMAP_PAIR_ID_DECODING_CONSTANT = 2147483647


class ColmapImageRetriever:
    """Class to handle COLMAP-based image retrieval"""
    
    def __init__(self, 
                cam1_folder: Path,
                cam2_folder: Path,
                cam1_camera_model: str,
                cam2_camera_model: str,
                colmap_workspace_dir: Path = None):
        """
        Initialize the image retriever.
        
        Args:
            cam1_folder: Path to cam1 images
            cam2_folder: Path to cam2 images
            cam1_camera_model: COLMAP camera model for cam1 (e.g., "OPENCV_FISHEYE", "SIMPLE_RADIAL")
            cam2_camera_model: COLMAP camera model for cam2
            colmap_workspace_dir: Directory for COLMAP workspace. Defaults to constants.COLMAP_DIR
        """
        self.colmap_workspace_dir = colmap_workspace_dir or constants.COLMAP_DIR
        self.colmap_workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Define camera configurations
        self.camera_configs = {
            "cam1": {
                "image_folder": cam1_folder,
                "camera_model": cam1_camera_model
            },
            "cam2": {
                "image_folder": cam2_folder,
                "camera_model": cam2_camera_model
            }
        }
        
        # Generate database paths dynamically based on folder names
        self.db_paths = {}
        for camera_name in self.camera_configs:
            # Create unique workspace based on camera and folder type
            folder_name = self.camera_configs[camera_name]["image_folder"].name
            workspace_suffix = f"{camera_name}_{folder_name}"
            db_path = self._get_database_path(workspace_suffix)
            self.db_paths[camera_name] = db_path
            
        print(f"Camera models - CAM1: {cam1_camera_model}, CAM2: {cam2_camera_model}")
    
    def _get_database_path(self, dataset_name: str) -> Path:
        """Get the database path for a given dataset name."""
        workspace_dir = self.colmap_workspace_dir / f"workspace_{dataset_name}"
        return workspace_dir / f"database_{dataset_name}.db"
    
    def initialize_databases(self, force_rebuild: bool = False):
        """
        Initialize COLMAP databases for all cameras if they don't exist.
        
        Args:
            force_rebuild: If True, rebuild databases even if they exist
        """
        for camera_name, config in self.camera_configs.items():
            db_path = self.db_paths[camera_name]
            
            if force_rebuild or not db_path.exists():
                print(f"Initializing {camera_name.upper()} database...")
                self._process_camera_dataset(
                    camera_name,
                    config["image_folder"],
                    config["camera_model"]
                )
            else:
                print(f"{camera_name.upper()} database already exists, skipping...")
    
    def _process_camera_dataset(self, 
                            dataset_name: str,
                            image_folder: Path,
                            camera_model: str) -> Optional[Path]:
        """
        Process a camera dataset with COLMAP feature extraction and matching.
        
        Returns:
            Path to the database if successful, None otherwise
        """
        workspace_dir = self.colmap_workspace_dir / f"workspace_{dataset_name}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        database_path = self._get_database_path(dataset_name)  # Use the same method!
        
        if not image_folder.exists():
            print(f"Error: Image folder {image_folder} does not exist")
            return None
            
        # Count images
        image_count = len(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))
        if image_count == 0:
            print(f"No images found in {image_folder}")
            return None
            
        print(f"Processing {image_count} images from {image_folder}")
        
        # Feature extraction
        print(f"[{dataset_name}] Running feature extraction...")
        cmd_feature_extraction = [
            COLMAP_EXE_PATH, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_folder),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.num_threads", "1"
        ]
        
        try:
            result = subprocess.run(cmd_feature_extraction, check=True, capture_output=True, text=True)
            print(f"[{dataset_name}] Feature extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"[{dataset_name}] Feature extraction failed: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"Error: COLMAP executable not found. Make sure 'colmap' is in your PATH")
            return None
        
        # Feature matching
        print(f"[{dataset_name}] Running feature matching...")
        cmd_feature_matching = [
            COLMAP_EXE_PATH, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "0",  # Set to "1" if you have GPU
            "--SiftMatching.num_threads", "1"
        ]
        
        try:
            result = subprocess.run(cmd_feature_matching, check=True, capture_output=True, text=True)
            print(f"[{dataset_name}] Feature matching completed")
        except subprocess.CalledProcessError as e:
            print(f"[{dataset_name}] Feature matching failed: {e}")
            print(f"Stderr: {e.stderr}")
            return None
            
        return database_path
    
    def find_matches_for_image(self, 
                            query_image_path: Path,
                            top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Find top matches for a query image.
        
        Args:
            query_image_path: Path to the query image
            top_n: Number of top matches to return
            
        Returns:
            List of tuples (matched_image_name, num_inliers)
        """
        query_image_path = Path(query_image_path)
        
        # Determine which camera based on the path
        dataset_name = None
        for camera_name in self.camera_configs:
            if camera_name in str(query_image_path).lower():
                dataset_name = camera_name
                break
        
        if not dataset_name:
            print(f"Error: Cannot determine camera from path {query_image_path}")
            return []
        
        db_path = self.db_paths[dataset_name]
        
        if not db_path.exists():
            print(f"Database for {dataset_name} not initialized. Run initialize_databases() first.")
            return []
        
        query_image_name = query_image_path.name
        return self._query_database(db_path, query_image_name, dataset_name, top_n)
    
    def _query_database(self,
                    database_path: Path,
                    query_image_name: str,
                    top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Query the COLMAP database for top matches.
        
        Returns:
            List of tuples (matched_image_name, num_inliers)
        """
        conn = None
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            # Get image_id for query image
            cursor.execute("SELECT image_id FROM images WHERE name = ?", (query_image_name,))
            result = cursor.fetchone()
            if not result:
                print(f"Image '{query_image_name}' not found in database")
                return []
            
            query_image_id = result[0]
            
            # Query for matches
            query_sql = f"""
                SELECT
                    CASE
                        WHEN (tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ? THEN i2.name
                        ELSE i1.name
                    END AS matched_image_name,
                    tvg.rows AS num_inliers
                FROM
                    two_view_geometries AS tvg
                JOIN
                    images AS i1 ON i1.image_id = (tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT})
                JOIN
                    images AS i2 ON i2.image_id = (tvg.pair_id % {COLMAP_PAIR_ID_DECODING_CONSTANT})
                WHERE
                    ((tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ? OR 
                    (tvg.pair_id % {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ?)
                    AND tvg.config > 0
                ORDER BY
                    tvg.rows DESC
                LIMIT ?;
            """
            
            cursor.execute(query_sql, (query_image_id, query_image_id, query_image_id, top_n))
            matches = cursor.fetchall()
            
            return matches
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()


def retrieve_top_matches(query_image_path: Path, 
                        cam1_folder: Path,
                        cam2_folder: Path,
                        cam1_camera_model: str,
                        cam2_camera_model: str,
                        top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Main function to retrieve top matches for a query image.
    This can be called from main.py.
    
    Args:
        query_image_path: Path to the query image
        cam1_folder: Path to cam1 images
        cam2_folder: Path to cam2 images
        cam1_camera_model: COLMAP camera model for cam1 (e.g., "OPENCV_FISHEYE", "SIMPLE_RADIAL")
        cam2_camera_model: COLMAP camera model for cam2
        top_n: Number of top matches to return
        
    Returns:
        List of tuples (matched_image_name, num_inliers) sorted by inliers
    """
    retriever = ColmapImageRetriever(
        cam1_folder=cam1_folder,
        cam2_folder=cam2_folder,
        cam1_camera_model=cam1_camera_model,
        cam2_camera_model=cam2_camera_model
    )
    
    # Initialize databases if not already done
    retriever.initialize_databases()
    
    # Find matches
    matches = retriever.find_matches_for_image(query_image_path, top_n)
    
    if matches:
        print(f"\nTop {len(matches)} matches for {query_image_path.name}:")
        for i, (name, inliers) in enumerate(matches, 1):
            print(f"  {i}. {name} ({inliers} inliers)")
    else:
        print(f"No matches found for {query_image_path.name}")
    
    return matches


def main():
    """Main function for testing and batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test COLMAP image retrieval")
    parser.add_argument("--distorted", action="store_true", 
                    help="Use distorted images instead of undistorted")
    parser.add_argument("--rebuild", action="store_true",
                    help="Force rebuild of databases")
    args = parser.parse_args()
    
    # Choose folders and camera models based on command line argument
    if args.distorted:
        print("Using DISTORTED images for retrieval")
        cam1_folder = constants.CAM1_DISTORTED_FOLDER
        cam2_folder = constants.CAM2_DISTORTED_FOLDER
        cam1_model = "OPENCV_FISHEYE"  # For 200° FOV fisheye
        cam2_model = "OPENCV_FISHEYE"
    else:
        print("Using UNDISTORTED images for retrieval")
        cam1_folder = constants.CAM1_UNDISTORTED_FOLDER
        cam2_folder = constants.CAM2_UNDISTORTED_FOLDER
        cam1_model = "SIMPLE_RADIAL"  # For rectilinear images
        cam2_model = "SIMPLE_RADIAL"
    
    # Initialize retriever with explicit camera models
    retriever = ColmapImageRetriever(
        cam1_folder=cam1_folder,
        cam2_folder=cam2_folder,
        cam1_camera_model=cam1_model,
        cam2_camera_model=cam2_model
    )
    
    # Initialize databases
    print(f"Initializing COLMAP databases with camera models: {cam1_model}")
    retriever.initialize_databases(force_rebuild=args.rebuild)
    
    # Example: Find matches for a specific image
    test_images = list(cam1_folder.glob("*.jpg"))[:3]  # Test first 3 images
    
    for test_image in test_images:
        print(f"\n{'='*60}")
        matches = retriever.find_matches_for_image(test_image, top_n=5)
        
        if matches:
            print(f"\nTop matches for {test_image.name}:")
            for i, (name, inliers) in enumerate(matches, 1):
                print(f"  {i}. {name} ({inliers} inliers)")
        else:
            print(f"No matches found for {test_image.name}")
    
    return 0


if __name__ == "__main__":
    exit(main())