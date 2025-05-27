import subprocess
from pathlib import Path
import sqlite3
import config

COLMAP_EXE_PATH = "colmap" # Default to "colmap" command
COLMAP_PAIR_ID_DECODING_CONSTANT = 2097152

# --- Function to Get Top Matches for a Specific Query Image ---
def get_top_matches_for_query(database_path: Path,
                                camera_name: str,
                                query_image_name: str,
                                top_n: int = 5):
    """
    Queries the COLMAP database to find the top N images that match a specific query image,
    based on the number of geometrically verified inlier matches.

    Args:
        database_path: Path to the COLMAP SQLite database file.
        camera_name: Name of the camera (for logging).
        query_image_name: The filename of the query image (e.g., "image_001.jpg").
        top_n: The number of top matches to retrieve for the query image.

    Returns:
        A list of tuples, where each tuple is (matched_image_name, num_inliers).
        Returns an empty list if an error occurs, the query image isn't found,
        or no matches are found.
    """
    print(f"\n[ {camera_name} ] Retrieving top {top_n} matches for query image '{query_image_name}'...")
    conn = None
    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # First, get the image_id of the query image from the 'images' table
        cursor.execute("SELECT image_id FROM images WHERE name = ?", (query_image_name,))
        result = cursor.fetchone()
        if not result:
            print(f"[ {camera_name} ] Query image '{query_image_name}' not found in the database.")
            return []
        query_image_id = result[0]

        query = f"""
            SELECT
                CASE
                    WHEN i1.image_id = ? THEN i2.name
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
                (i1.image_id = ? OR i2.image_id = ?)
                AND tvg.config > 0
            ORDER BY
                tvg.rows DESC
            LIMIT ?;
        """
        cursor.execute(query, (query_image_id, query_image_id, query_image_id, top_n))
        top_matches = cursor.fetchall()

        if top_matches:
            print(f"[ {camera_name} ] Top {len(top_matches)} matches for '{query_image_name}':")
            for i, (matched_img, count) in enumerate(top_matches):
                print(f"  {i+1}. {query_image_name} <-> {matched_img} : {count} inliers")
        else:
            print(f"[ {camera_name} ] No geometrically verified matches found for '{query_image_name}'.")
        return top_matches

    except sqlite3.Error as e:
        print(f"[ {camera_name} ] SQLite error while retrieving matches for query '{query_image_name}': {e}")
        return []
    finally:
        if conn:
            conn.close()

# --- Main Colmap Processing Function ---
def extract_and_match_image_features(camera_name: str,
                                image_dir: Path,
                                base_workspace_dir: Path, # This is config.COLMAP_DIR
                                ):

    workspace_dir = base_workspace_dir / f"workspace_{camera_name}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / f"database_{camera_name}.db"

    print(f"\n[ {camera_name} ] Running Feature Extraction...")
    cmd_feature_extraction = [
        COLMAP_EXE_PATH, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "OPENCV_FISHEYE",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ]
    try:
        subprocess.run(cmd_feature_extraction, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Feature extraction successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error during feature extraction: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: Colmap executable '{COLMAP_EXE_PATH}' not found. Is it in PATH?")
        return None

    print(f"\n[ {camera_name} ] Running Feature Matching...")
    cmd_feature_matching = [
        COLMAP_EXE_PATH, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0",
    ]
    try:
        subprocess.run(cmd_feature_matching, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Feature matching successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error during feature matching: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None
    
    return database_path

# --- Main Script Logic ---
if __name__ == "__main__":
    config.COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    # * Change query images here, can be done with function later.
    query_image_name_cam1 = config.COLMAP_DIR / "images_cam1/cam1_0801.jpg"
    query_image_name_cam2 = config.DATASET1_DIR / "images_cam2/cam2_0801.jpg"

    # Process Camera 1
    print(f"\n--- Starting processing for Camera 1 ---")
    cam1_image_folder = config.COLMAP_DIR / "images_cam1"
    db_path_cam1 = None
    if cam1_image_folder.is_dir(): 
        db_path_cam1 = extract_and_match_image_features(
            "cam1", 
            cam1_image_folder, 
            config.COLMAP_DIR
        )
        if db_path_cam1:
            if query_image_name_cam1: # Ensure a query image name is provided
                top_matches_cam1 = get_top_matches_for_query(
                    db_path_cam1,
                    "cam1",
                    query_image_name_cam1,
                    top_n=5
                )
                if top_matches_cam1:
                    print(f"\n[ cam1 ] Retrieved top matches for '{query_image_name_cam1}'.")

    # Process Camera 2
    print(f"\n--- Starting processing for Camera 2 ---")
    cam2_image_folder = config.COLMAP_DIR / "images_cam2"
    db_path_cam2 = None
    if cam2_image_folder.is_dir():
        db_path_cam2 = extract_and_match_image_features(
            "cam2", 
            cam2_image_folder, 
            config.COLMAP_DIR
        )
        if db_path_cam2:
            if query_image_name_cam2: # Ensure a query image name is provided
                top_matches_cam2 = get_top_matches_for_query(
                    db_path_cam2,
                    "cam2",
                    query_image_name_cam2,
                    top_n=5
                )
                if top_matches_cam2:
                    print(f"\n[ cam2 ] Retrieved top matches for '{query_image_name_cam2}'.")
    
    print("\n--- Processing complete ---")