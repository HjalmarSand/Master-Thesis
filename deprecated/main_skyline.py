# main.py
import sys
print(f"Python version executing this script: {sys.version}")
import os
import glob
from PIL import Image
import numpy as np

try:
    from image_segmenter import ImageSegmenter
    from skyline_extractor import SkylineExtractor
    from skyline_database import SkylineDatabase
    from skyline_matcher import SkylineMatcher
    from visualizer import Visualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure segmenter.py, skyline_extractor.py, skyline_database.py, skyline_matcher.py, and visualizer.py are in the same directory or Python path.")
    exit()

# --- Configuration ---
# Model for segmentation (ensure it has a 'sky' class)
SEGMENTATION_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
SKY_LABEL = "sky" # The label name for sky in the chosen model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATABASE_DIR = os.path.join(BASE_DIR, "dataset1", "split_images")
IMAGE_QUERY_DIR = os.path.join(BASE_DIR,"dataset1/sample_images")
SKYLINE_DB_FILE = os.path.join(BASE_DIR,"skyline_db.json")

os.makedirs(IMAGE_DATABASE_DIR, exist_ok=True)
os.makedirs(IMAGE_QUERY_DIR, exist_ok=True)



def process_single_image(image_path: str, 
                        segmenter: ImageSegmenter, 
                        extractor: SkylineExtractor,
                        visualizer: Visualizer = None,
                        show_visualization: bool = False) -> np.ndarray | None:
    """
    Processes a single image: segments sky, extracts skyline.
    Optionally visualizes the steps.
    """
    print(f"\nProcessing image: {image_path}")
    
    #Sky Mask
    sky_mask = segmenter.get_sky_mask(image_path)
    if sky_mask is None or not np.any(sky_mask): # Check if mask is valid a
        print(f"Could not get a valid sky mask for {image_path}.")
        if visualizer and show_visualization:
            try:
                img_pil = Image.open(image_path)
                visualizer.display_segmentation_results(img_pil, sky_mask_np=sky_mask, title_prefix=f"{os.path.basename(image_path)} - ")
            except FileNotFoundError:
                pass
        return None

    #Extract Skyline
    skyline_vector = extractor.extract_skyline(sky_mask)
    if skyline_vector is None or skyline_vector.size == 0:
        print(f"Could not extract skyline for {image_path}.")
        if visualizer and show_visualization:
            visualizer.display_segmentation_results(image_path, sky_mask_np=sky_mask, title_prefix=f"{os.path.basename(image_path)} - ")
        return None

    print(f"Successfully extracted skyline for {image_path} (length: {len(skyline_vector)})")

    #(Optional) Visualize
    if visualizer and show_visualization:
        visualizer.display_segmentation_results(
            image_path, 
            sky_mask_np=sky_mask, 
            skyline_vector=skyline_vector,
            title_prefix=f"{os.path.basename(image_path)} - "
        )
    return skyline_vector


def populate_database(image_dir: str, 
                    skyline_db: SkylineDatabase, 
                    segmenter: ImageSegmenter, 
                    extractor: SkylineExtractor,
                    visualizer: Visualizer = None,
                    show_intermediate_viz: bool = False,
                    overwrite_existing: bool = False):
    """
    Processes all images in a directory and adds their skylines to the database.
    """
    print(f"\n--- Populating Skyline Database from: {image_dir} ---")
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"No images found in {image_dir}.")
        return

    processed_count = 0
    for image_path in image_paths:
        image_id = os.path.basename(image_path)
        
        if not overwrite_existing and skyline_db.get_skyline_data(image_id) is not None:
            print(f"Skyline for {image_id} already in database. Skipping.")
            continue

        skyline_vec = process_single_image(image_path, segmenter, extractor, visualizer, show_intermediate_viz)
        
        if skyline_vec is not None:
            skyline_db.add_skyline(image_id, skyline_vec, metadata={"original_path": image_path})
            processed_count += 1
        else:
            print(f"Failed to process {image_id} for database.")
            
    print(f"\n--- Database Population Complete ---")
    print(f"Processed {processed_count} new images.")
    print(f"Total skylines in database: {skyline_db.count()}")
    if processed_count > 0 or (overwrite_existing and image_paths):
        skyline_db.save_database()


def query_skyline(query_image_path: str,
                    skyline_db: SkylineDatabase,
                    segmenter: ImageSegmenter,
                    extractor: SkylineExtractor,
                    matcher: SkylineMatcher,
                    visualizer: Visualizer,
                    top_n: int = 3,
                    show_query_viz: bool = True,
                    show_match_viz: bool = True):
    """
    Processes a query image, extracts its skyline, and finds matches in the database.
    """
    print(f"\n--- Querying with image: {query_image_path} ---")
    
    query_skyline_vec = process_single_image(query_image_path, segmenter, extractor, visualizer, show_query_viz)

    if query_skyline_vec is None:
        print("Cannot perform query: failed to extract skyline from query image.")
        return

    all_db_skylines = skyline_db.get_all_skylines_with_ids()
    if not all_db_skylines:
        print("Database is empty. Cannot find matches.")
        return

    print(f"Query skyline length: {len(query_skyline_vec)}")

    matches = matcher.find_matches(query_skyline_vec, all_db_skylines, top_n=top_n)

    print(f"\nTop {top_n} matches for {os.path.basename(query_image_path)}:")
    if not matches:
        print("No matches found.")
        return

    for i, (image_id, score) in enumerate(matches):
        print(f"{i+1}. Image ID: {image_id}, Score: {score:.4f}")
        db_entry = skyline_db.get_skyline_data(image_id)
        if db_entry and show_match_viz and visualizer:
            match_image_path = db_entry["metadata"].get("original_path", os.path.join(IMAGE_DATABASE_DIR, image_id))
            if os.path.exists(match_image_path):
                print(f"   Displaying match: {match_image_path}")
                visualizer.display_segmentation_results(
                    match_image_path, 
                    skyline_vector=db_entry["vector"], 
                    title_prefix=f"Match {i+1}: {image_id} (Score: {score:.2f}) - "
                )
            else:
                print(f"   Original image for match {image_id} not found at {match_image_path}")

if __name__ == "__main__":
    print("Initializing components...")
    segmenter = ImageSegmenter(model_name=SEGMENTATION_MODEL_NAME, sky_label_name=SKY_LABEL)
    extractor = SkylineExtractor()
    database = SkylineDatabase(db_file_path=SKYLINE_DB_FILE) # Loads if exists
    matcher = SkylineMatcher(metric='mse') # Use 'mse' or 'mae'
    visualizer = Visualizer()


    #Populate the database from images in IMAGE_DATABASE_DIR ---
    #Set show_intermediate_viz=True to see processing for each DB image
    #Set overwrite_existing=True to re-process all images
    populate_database(
        IMAGE_DATABASE_DIR, 
        database, 
        segmenter, 
        extractor, 
        visualizer,
        show_intermediate_viz=False, 
        overwrite_existing=False     
    )

    #Query with an image from IMAGE_QUERY_DIR ---
    #List query images
    query_image_paths = glob.glob(os.path.join(IMAGE_QUERY_DIR, "*.jpg"))
    if not query_image_paths:
        print(f"\nNo query images found in {IMAGE_QUERY_DIR}. Add some .png, .jpg, or .jpeg files.")
    else:
        test_query_image_path = query_image_paths[0] 
        
        if os.path.exists(test_query_image_path):
            query_skyline(
                test_query_image_path,
                database,
                segmenter,
                extractor,
                matcher,
                visualizer,
                top_n=3,
                show_query_viz=True, # Show processing of the query image
                show_match_viz=True  # Show matched images from DB
            )
        else:
            print(f"Query image not found: {test_query_image_path}")

    print("\n--- Main script finished ---")