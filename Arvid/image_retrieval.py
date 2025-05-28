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
        top_n: The number of top pairs to retrieve for the query image.
    Returns:
        A list of tuples, where each tuple is (matched_image_name, num_inliers).
        Returns an empty list if an error occurs, the query image isn't found,
        or no matches are found.
    """
    conn = None
    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        cursor.execute("SELECT image_id FROM images WHERE name = ?", (query_image_name,))
        result = cursor.fetchone()
        if not result:
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
            print(f"[ {camera_name} ] Query: {query_image_name} -> Found {len(top_matches)} verified matches. Top one: {top_matches[0][0]} ({top_matches[0][1]} inliers)")
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
                                base_workspace_dir: Path,
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

    # Process Camera 1
    print(f"\n--- Starting processing for Camera 1 ---")
    cam1_image_folder = config.COLMAP_DIR / "images_cam1"
    db_path_cam1 = None
    images_with_verified_matches_cam1 = []
    all_top_matches_cam1 = {} 

    if cam1_image_folder.is_dir(): 
        db_path_cam1 = extract_and_match_image_features(
            "cam1", 
            cam1_image_folder, 
            config.COLMAP_DIR
        )
        if db_path_cam1:
            print(f"\n[ cam1 ] Processing all images from folder '{cam1_image_folder}' for top matches...")
            
            input_image_names_cam1 = sorted([
                f.name for f in cam1_image_folder.iterdir() 
                if f.is_file()
            ])

            if not input_image_names_cam1:
                print(f"[ cam1 ] No image files found in folder {cam1_image_folder}")
            else:
                print(f"[ cam1 ] Found {len(input_image_names_cam1)} images in the input folder. Querying each...")
                for image_filename in input_image_names_cam1:
                    top_matches = get_top_matches_for_query(
                        db_path_cam1,
                        "cam1",
                        image_filename,
                        top_n=5
                    )
                    if top_matches: 
                        images_with_verified_matches_cam1.append(image_filename)
                        all_top_matches_cam1[image_filename] = top_matches 
                
                print(f"\n[ cam1 ] Total images with at least one geometrically verified match: {len(images_with_verified_matches_cam1)}")
                if images_with_verified_matches_cam1:
                    print(f"[ cam1 ] Images with verified matches: {images_with_verified_matches_cam1}")
                else:
                    print(f"[ cam1 ] No images had any geometrically verified matches.")

    # Process Camera 2
    print(f"\n--- Starting processing for Camera 2 ---")
    cam2_image_folder = config.COLMAP_DIR / "images_cam2" # Source of truth for image names
    db_path_cam2 = None
    images_with_verified_matches_cam2 = []
    all_top_matches_cam2 = {} 

    if cam2_image_folder.is_dir():
        db_path_cam2 = extract_and_match_image_features(
            "cam2", 
            cam2_image_folder, 
            config.COLMAP_DIR
        )
        if db_path_cam2:
            print(f"\n[ cam2 ] Processing all images from folder '{cam2_image_folder}' for top matches...")

            input_image_names_cam2 = sorted([
                f.name for f in cam2_image_folder.iterdir()
                if f.is_file()
            ])

            if not input_image_names_cam2:
                print(f"[ cam2 ] No image files found in folder {cam2_image_folder}")
            else:
                print(f"[ cam2 ] Found {len(input_image_names_cam2)} images in the input folder. Querying each...")
                for image_filename in input_image_names_cam2:
                    top_matches = get_top_matches_for_query(
                        db_path_cam2,
                        "cam2",
                        image_filename,
                        top_n=5
                    )
                    if top_matches: 
                        images_with_verified_matches_cam2.append(image_filename)
                        all_top_matches_cam2[image_filename] = top_matches

                print(f"\n[ cam2 ] Total images with at least one geometrically verified match: {len(images_with_verified_matches_cam2)}")
                if images_with_verified_matches_cam2:
                    print(f"[ cam2 ] Images with verified matches: {images_with_verified_matches_cam2}")
                else:
                    print(f"[ cam2 ] No images had any geometrically verified matches.")
    
    print("\n--- Processing complete ---")