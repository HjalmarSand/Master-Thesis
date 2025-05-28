import subprocess
from pathlib import Path
import sqlite3
import config

COLMAP_EXE_PATH = "colmap"
COLMAP_PAIR_ID_DECODING_CONSTANT = 2147483647

def debug_database_contents(database_path: Path, dataset_name: str):
    """Debug function to inspect the actual database contents"""
    conn = None
    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        
        print(f"\n[DEBUG {dataset_name}] Database inspection:")
        
        # Check images table
        cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
        images = cursor.fetchall()
        print(f"[DEBUG {dataset_name}] Images in database:")
        for img in images:
            print(f"  image_id={img[0]}, name={img[1]}")
        
        # Check two_view_geometries table structure and contents
        cursor.execute("PRAGMA table_info(two_view_geometries)")
        columns = cursor.fetchall()
        print(f"\n[DEBUG {dataset_name}] two_view_geometries table structure:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Check all entries in two_view_geometries
        cursor.execute("SELECT pair_id, rows, config FROM two_view_geometries ORDER BY pair_id")
        geometries = cursor.fetchall()
        print(f"\n[DEBUG {dataset_name}] All entries in two_view_geometries:")
        for geo in geometries:
            # Try to decode pair_id
            img1_id = geo[0] // COLMAP_PAIR_ID_DECODING_CONSTANT
            img2_id = geo[0] % COLMAP_PAIR_ID_DECODING_CONSTANT
            print(f"  pair_id={geo[0]} (decoded: {img1_id}-{img2_id}), rows={geo[1]}, config={geo[2]}")
        
        # Check specifically for entries with config > 0
        cursor.execute("SELECT pair_id, rows, config FROM two_view_geometries WHERE config > 0 ORDER BY pair_id")
        valid_geometries = cursor.fetchall()
        print(f"\n[DEBUG {dataset_name}] Entries with config > 0:")
        for geo in valid_geometries:
            img1_id = geo[0] // COLMAP_PAIR_ID_DECODING_CONSTANT
            img2_id = geo[0] % COLMAP_PAIR_ID_DECODING_CONSTANT
            print(f"  pair_id={geo[0]} (decoded: {img1_id}-{img2_id}), rows={geo[1]}, config={geo[2]}")
            
    except sqlite3.Error as e:
        print(f"[DEBUG {dataset_name}] SQLite error: {e}")
    finally:
        if conn:
            conn.close()

def get_top_matches_for_query(database_path: Path,
                                dataset_name_for_log: str,
                                query_image_name: str,
                                top_n: int = 5):
    conn = None
    print(f"\n[QUERY {dataset_name_for_log}] For Query Image Name: {query_image_name}")
    try:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # Get the image_id for the current query_image_name
        cursor.execute("SELECT image_id FROM images WHERE name = ?", (query_image_name,))
        result = cursor.fetchone()
        if not result:
            print(f"[QUERY {dataset_name_for_log}] Image name '{query_image_name}' NOT FOUND in 'images' table.")
            return []
        
        current_query_image_id = result[0]
        print(f"[QUERY {dataset_name_for_log}] Image name '{query_image_name}' has image_id = {current_query_image_id}")

        query_sql = f"""
            SELECT
                CASE
                    WHEN (tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ? THEN i2.name
                    ELSE i1.name
                END AS matched_image_name,
                tvg.rows AS num_inliers,
                tvg.pair_id,
                (tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT}) AS img1_id,
                (tvg.pair_id % {COLMAP_PAIR_ID_DECODING_CONSTANT}) AS img2_id
            FROM
                two_view_geometries AS tvg
            JOIN
                images AS i1 ON i1.image_id = (tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT})
            JOIN
                images AS i2 ON i2.image_id = (tvg.pair_id % {COLMAP_PAIR_ID_DECODING_CONSTANT})
            WHERE
                ((tvg.pair_id / {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ? OR (tvg.pair_id % {COLMAP_PAIR_ID_DECODING_CONSTANT}) = ?)
                AND tvg.config > 0
            ORDER BY
                tvg.rows DESC
            LIMIT ?;
        """
        
        print(f"[QUERY {dataset_name_for_log}] Executing SQL query for image_id {current_query_image_id}")
        cursor.execute(query_sql, (current_query_image_id, current_query_image_id, current_query_image_id, top_n))
        top_matches = cursor.fetchall()

        if top_matches:
            print(f"[QUERY {dataset_name_for_log}] Found {len(top_matches)} verified matches:")
            for i, match in enumerate(top_matches):
                print(f"  {i+1}. {match[0]} ({match[1]} inliers, pair_id={match[2]}, decoded as {match[3]}-{match[4]})")
            return [(match[0], match[1]) for match in top_matches]  # Return only name and inliers
        else:
            print(f"[QUERY {dataset_name_for_log}] No verified matches found.")
            return []

    except sqlite3.Error as e:
        print(f"[QUERY {dataset_name_for_log}] SQLite error for query '{query_image_name}': {e}")
        return []
    finally:
        if conn:
            conn.close()

def extract_and_match_image_features(dataset_name: str,
                                    image_dir: Path,
                                    base_workspace_dir: Path,
                                    camera_model: str
                                    ):
    workspace_dir = base_workspace_dir / f"workspace_{dataset_name}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / f"database_{dataset_name}.db"

    print(f"\n[ {dataset_name} ] Running Feature Extraction...")
    cmd_feature_extraction = [
        COLMAP_EXE_PATH, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0", # ? Change this later if on home pc. 1 for gpu
        "--SiftExtraction.num_threads", "1" # ? Change this later if running on more powerful machine.
    ]
    try:
        subprocess.run(cmd_feature_extraction, check=True, capture_output=True, text=True)
        print(f"[ {dataset_name} ] Feature extraction successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {dataset_name} ] Error during feature extraction: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: Colmap executable '{COLMAP_EXE_PATH}' not found. Is it in PATH?")
        return None

    print(f"\n[ {dataset_name} ] Running Feature Matching...")
    cmd_feature_matching = [
        COLMAP_EXE_PATH, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0", # ? Change this later if on home pc. 1 for gpu
    ]
    try:
        subprocess.run(cmd_feature_matching, check=True, capture_output=True, text=True)
        print(f"[ {dataset_name} ] Feature matching successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {dataset_name} ] Error during feature matching: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None
    
    return database_path

def process_dataset_and_find_matches(dataset_name: str, 
                                    image_folder_name_relative_to_colmap_dir: str, 
                                    colmap_base_dir: Path,
                                    camera_model: str
                                    ):
    """
    Processes a single dataset: runs COLMAP, then finds and reports images with matches.
    """
    print(f"\n--- Starting processing for Dataset: {dataset_name} ---")
    
    image_folder = colmap_base_dir / image_folder_name_relative_to_colmap_dir
    
    if not image_folder.is_dir():
        print(f"[ {dataset_name} ] Image folder '{image_folder}' not found. Skipping.")
        return [], {}

    db_path = extract_and_match_image_features(
        dataset_name, 
        image_folder, 
        colmap_base_dir,
        camera_model
    )

    images_with_verified_matches_list = []
    all_top_matches_dict = {} 

    if db_path:
        # ? Add debug inspection?
        # debug_database_contents(db_path, dataset_name)
        
        print(f"\n[ {dataset_name} ] Processing all images from folder '{image_folder}' for top matches...")
        
        input_image_filenames = sorted([
            f.name for f in image_folder.glob("*.jpg")
            if f.is_file()
        ])

        if not input_image_filenames:
            print(f"[ {dataset_name} ] No .jpg image files found in folder {image_folder}")
        else:
            print(f"[ {dataset_name} ] Found {len(input_image_filenames)} .jpg images in the input folder. Querying each...")
            for image_filename in input_image_filenames:
                top_matches = get_top_matches_for_query(
                    db_path,
                    dataset_name,
                    image_filename,
                    top_n=5
                )
                if top_matches: 
                    images_with_verified_matches_list.append(image_filename)
                    all_top_matches_dict[image_filename] = top_matches
            
            print(f"\n[ {dataset_name} ] Total images with at least one geometrically verified match: {len(images_with_verified_matches_list)}")
            if images_with_verified_matches_list:
                print(f"[ {dataset_name} ] Images with verified matches: {images_with_verified_matches_list}")
            else:
                print(f"[ {dataset_name} ] No images had any geometrically verified matches.")
    
    return images_with_verified_matches_list, all_top_matches_dict

# --- Main Script Logic ---
if __name__ == "__main__":
    if not hasattr(config, 'COLMAP_DIR') or not isinstance(config.COLMAP_DIR, Path):
        print("Error: config.py must define COLMAP_DIR as a pathlib.Path pointing to your project root.")
        exit(1)
        
    colmap_project_root = config.COLMAP_DIR
    colmap_project_root.mkdir(parents=True, exist_ok=True)

    # --- Process Camera 1 ---
    images_with_verified_matches_cam1, all_top_matches_cam1 = process_dataset_and_find_matches(
        dataset_name="cam1",
        image_folder_name_relative_to_colmap_dir="images_cam1",
        colmap_base_dir=colmap_project_root,
        camera_model = "OPENCV_FISHEYE"
    )

    # --- Process Camera 2 ---
    images_with_verified_matches_cam2, all_top_matches_cam2 = process_dataset_and_find_matches(
        dataset_name="cam2",
        image_folder_name_relative_to_colmap_dir="images_cam2",
        colmap_base_dir=colmap_project_root,
        camera_model = "OPENCV_FISHEYE"
    )
    
    # --- Process Test_Set ---
    images_with_verified_matches_test_set, all_top_matches_test_set = process_dataset_and_find_matches(
        dataset_name="Test_Set",
        image_folder_name_relative_to_colmap_dir="Test_Set",
        colmap_base_dir=colmap_project_root,
        camera_model="SIMPLE_RADIAL"
    )
    
    # print("\n--- Processing complete ---")
    # print("\n--- Summary of Results ---")
    # print(f"Test_Set images with matches: {images_with_verified_matches_test_set}")
    
    # Print detailed matches for debugging
    # if all_top_matches_test_set:
    #     print("\n--- Detailed Matches ---")
    #     for image_name, matches in all_top_matches_test_set.items():
    #         print(f"{image_name}:")
    #         for match_name, inliers in matches:
    #             print(f"  -> {match_name} ({inliers} inliers)")