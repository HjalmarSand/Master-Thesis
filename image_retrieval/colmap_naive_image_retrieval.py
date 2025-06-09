# Image retrieval using colmap that internally uses sift then ann matcher, then geometric verification to find top matches.
# See publication below detailing the methods. This file uses the naive matcher so far, and is infeasible for large datasets,
# but can be used as a baseline that we later compare with speeding up lookup with BoW, probably with MegaLoc that seems promising
# in its description. 
# https://www.researchgate.net/publication/301197096_Structure-from-Motion_Revisited

import subprocess
from pathlib import Path
import sqlite3
from typing import List, Tuple, Optional

import constants

COLMAP_EXE_PATH = "colmap"
COLMAP_PAIR_ID_DECODING_CONSTANT = 2147483647


class ColmapImageRetriever:
    def __init__(self, 
                cam1_folder: Path,
                cam2_folder: Path,
                cam1_camera_model: str,
                cam2_camera_model: str,
                # Add type hints for clarity
                cam1_type: str, # e.g., "distorted" or "undistorted"
                cam2_type: str, # e.g., "distorted" or "undistorted"
                colmap_workspace_dir: Path = None):

        self.colmap_workspace_dir = colmap_workspace_dir or constants.COLMAP_DIR
        self.colmap_workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.camera_configs = {
            "cam1": {
                "image_folder": cam1_folder,
                "camera_model": cam1_camera_model,
                "type": cam1_type
            },
            "cam2": {
                "image_folder": cam2_folder,
                "camera_model": cam2_camera_model,
                "type": cam2_type
            }
        }
        
        self.db_paths = {}
        for camera_name, config in self.camera_configs.items():
            # Use camera_name and its type for a unique suffix
            workspace_suffix = f"{camera_name}_{config['type']}" 
            db_path = self._get_database_path(workspace_suffix)
            self.db_paths[workspace_suffix] = db_path
            
        print(f"Camera models - CAM1 ({cam1_type}): {cam1_camera_model}, CAM2 ({cam2_type}): {cam2_camera_model}")
    
    def _get_database_path(self, workspace_suffix: str) -> Path:
        """Get the database path for a given workspace_suffix."""
        workspace_dir = self.colmap_workspace_dir / f"workspace_{workspace_suffix}"
        return workspace_dir / f"database_{workspace_suffix}.db"
    
    def initialize_databases(self, force_rebuild: bool = False):
        for camera_name, config in self.camera_configs.items():
            workspace_suffix = f"{camera_name}_{config['type']}"
            db_path = self.db_paths[workspace_suffix]
            
            if force_rebuild or not db_path.exists():
                print(f"Initializing {workspace_suffix.upper()} database...")
                self._process_camera_dataset(
                    workspace_suffix,
                    config["image_folder"],
                    config["camera_model"]
                )
            else:
                print(f"{workspace_suffix.upper()} database already exists, skipping...")
    
    def _process_camera_dataset(self, 
                            workspace_suffix: str,
                            image_folder: Path,
                            camera_model: str) -> Optional[Path]:
        
        workspace_dir = self.colmap_workspace_dir / f"workspace_{workspace_suffix}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        database_path = self._get_database_path(workspace_suffix)
        
        if not image_folder.exists():
            print(f"Error: Image folder {image_folder} does not exist")
            return None
            
        image_count = len(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))
        if image_count == 0:
            print(f"No images found in {image_folder}")
            return None
            
        print(f"Processing {image_count} images from {image_folder} for {workspace_suffix}")
        
        print(f"[{workspace_suffix}] Running feature extraction...")
        cmd_feature_extraction = [
            COLMAP_EXE_PATH, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_folder),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.num_threads", "1" # Set to 1 because my laptop cant handle more reliably
        ]
        
        try:
            subprocess.run(cmd_feature_extraction, check=True, capture_output=True, text=True)
            print(f"[{workspace_suffix}] Feature extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"[{workspace_suffix}] Feature extraction failed: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"Error: COLMAP executable not found. Make sure 'colmap' is in your PATH")
            return None
        
        print(f"[{workspace_suffix}] Running feature matching...")
        cmd_feature_matching = [
        COLMAP_EXE_PATH, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0", # ? Change this later if on home pc. 1 for gpu
        "--SiftMatching.num_threads", "1" # Only one because of crashes. Not sure why, it works well in gui
        ]
        
        try:
            subprocess.run(cmd_feature_matching, check=True, capture_output=True, text=True)
            print(f"[{workspace_suffix}] Feature matching completed")
        except subprocess.CalledProcessError as e:
            print(f"[{workspace_suffix}] Feature matching failed: {e}")
            print(f"Stderr: {e.stderr}")
            return None
            
        return database_path
    
    def find_matches_for_image(self, 
                            query_image_path: Path,
                            query_image_type: str, # "distorted" or "undistorted" for example
                            top_n: int = 5) -> List[Tuple[str, int]]:
        query_image_path = Path(query_image_path)
        
        camera_name_from_path = None
        for cam_name_key in self.camera_configs:
            # A simple check if "cam1" or "cam2" is in the parent folder name or image name
            # This might need to be more robust depending on exact naming
            if cam_name_key in str(query_image_path.parent.name).lower() or \
                cam_name_key in query_image_path.name.lower() :
                camera_name_from_path = cam_name_key
                break
        
        if not camera_name_from_path:
            print(f"Error: Cannot determine camera (cam1/cam2) from path {query_image_path}")
            return []

        workspace_suffix = f"{camera_name_from_path}_{query_image_type}"
        
        if workspace_suffix not in self.db_paths:
            print(f"Error: No database configuration found for {workspace_suffix}")
            return []
            
        db_path = self.db_paths[workspace_suffix]
        
        if not db_path.exists():
            print(f"Database for {workspace_suffix} not initialized. Run initialize_databases() first.")
            return []
        
        query_image_name = query_image_path.name
        return self._query_database(db_path, query_image_name, top_n) 
    
    def _query_database(self,
                    database_path: Path,
                    query_image_name: str,
                    top_n: int = 5) -> List[Tuple[str, int]]:
        conn = None
        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT image_id FROM images WHERE name = ?", (query_image_name,))
            result = cursor.fetchone()
            if not result:
                print(f"Image '{query_image_name}' not found in database {database_path}")
                return []
            
            query_image_id = result[0]
            
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
            print(f"Database error for {database_path}: {e}")
            return []
        finally:
            if conn:
                conn.close()

def retrieve_top_matches(query_image_path: Path, 
                        cam1_folder: Path,
                        cam2_folder: Path,
                        cam1_camera_model: str,
                        cam2_camera_model: str,
                        cam1_type: str,
                        cam2_type: str,
                        query_image_type: str,
                        top_n: int = 5) -> List[Tuple[str, int]]:
    retriever = ColmapImageRetriever(
        cam1_folder=cam1_folder,
        cam2_folder=cam2_folder,
        cam1_camera_model=cam1_camera_model,
        cam2_camera_model=cam2_camera_model,
        cam1_type=cam1_type,
        cam2_type=cam2_type
    )
    
    retriever.initialize_databases() # Consider passing force_rebuild if needed
    
    matches = retriever.find_matches_for_image(query_image_path, query_image_type, top_n)
    
    if matches:
        print(f"\nTop {len(matches)} matches for {query_image_path.name} (type: {query_image_type}):")
        for i, (name, inliers) in enumerate(matches, 1):
            print(f"  {i}. {name} ({inliers} inliers)")
    else:
        print(f"No matches found for {query_image_path.name} (type: {query_image_type})")
    
    return matches