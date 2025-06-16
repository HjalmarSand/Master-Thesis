# main_retrieval.py
import os
import pickle
import math
import csv
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from create_megaloc import MegaLocDatabaseCreator  # Database creation class

DATABASE_FILE = "megaloc_undist_db.pkl"
TOP_K_RESULTS = 5
MEGALOC_INPUT_IMAGE_SIZE = (480, 640)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Functions for Single-Query Mode (External Image) ---

def initialize_model_and_transforms():
    """Initialize MegaLoc model and transforms for processing an external image."""
    print("Initializing MegaLoc model...")
    try:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    except Exception as e:
        print(f"Error loading MegaLoc model from torch.hub: {e}"); exit()
    model.eval().to(device)
    preprocess = T.Compose([
        T.Resize(MEGALOC_INPUT_IMAGE_SIZE), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print("MegaLoc model and transforms initialized.")
    return model, preprocess

def extract_features(image_path, model, preprocess_fn):
    """Extract features from an external query image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess_fn(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t)
        return features.squeeze().cpu().numpy()
    except FileNotFoundError:
        print(f"Error: Query image not found at {image_path}"); return None
    except Exception as e:
        print(f"Error processing query image {image_path}: {e}"); return None

# --- Core Retrieval and Database Functions ---

def load_and_filter_database(database_path=None, all_database_entries=None, exclude_range_start=None, exclude_range_end=None):
    """Load the database from path or use pre-loaded list, and filter entries."""
    if all_database_entries is None:
        if database_path is None: raise ValueError("Must provide 'database_path' or 'all_database_entries'.")
        try:
            with open(database_path, 'rb') as f: all_database_entries = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Database file not found at {database_path}."); return None, None
        except Exception as e:
            print(f"Error loading database: {e}"); return None, None
    if not all_database_entries:
        print("Warning: Source database is empty."); return None, None

    filtered_entries = []
    for entry in all_database_entries:
        if exclude_range_start is not None and exclude_range_end is not None:
            try:
                numeric_part = int(entry["base_name"].split('.')[0])
                if exclude_range_start <= numeric_part <= exclude_range_end: continue
            except ValueError: pass
        filtered_entries.append(entry)

    if not filtered_entries: return None, None
    db_feature_matrix = np.array([item["features"] for item in filtered_entries])
    return filtered_entries, db_feature_matrix

def retrieve_similar_images(query_features, db_entries, db_feature_matrix, top_k=5):
    """Retrieve most similar images from the database."""
    if query_features is None or db_feature_matrix is None or db_feature_matrix.shape[0] == 0: return []
    similarities = cosine_similarity(query_features.reshape(1, -1), db_feature_matrix).flatten()
    top_k_indices = np.argsort(-similarities)[:top_k]
    # Add similarity score to the result dictionary
    results = []
    for i in top_k_indices:
        res = db_entries[i].copy()
        res['similarity'] = similarities[i]
        results.append(res)
    return results

def ensure_database_exists(db_path, create_new=False):
    """Ensure database exists, creating it if necessary."""
    db_creator = MegaLocDatabaseCreator(database_output_file=db_path)
    if create_new or not db_creator.database_exists():
        print(f"Creating new database at {db_path}...")
        if not db_creator.create_database(cameras=['CAM1', 'CAM2'], force_recreate=create_new):
            print("Failed to create database!"); return False
    else:
        print(f"Using existing database at {db_path}")
        info = db_creator.get_database_info()
        if info: print(f"Database contains {info['total_entries']} entries from cameras: {list(info['cameras'].keys())}")
    return True

# --- Evaluation and Mode-Specific Logic ---

def calculate_distance(entry1, entry2):
    """Calculates the Euclidean distance between two database entries."""
    return math.sqrt((entry1['easting'] - entry2['easting'])**2 + (entry1['northing'] - entry2['northing'])**2)

def run_single_query_mode(args):
    """Handles logic for querying with a single, external image file."""
    model, preprocess_fn = initialize_model_and_transforms()
    database_entries, db_feature_matrix = load_and_filter_database(database_path=args.db_file)
    if not database_entries: print("Database is empty. Cannot proceed."); return
    
    print(f"\nProcessing query image: {args.query_image}")
    query_features = extract_features(args.query_image, model, preprocess_fn)
    if query_features is None: return

    results = retrieve_similar_images(query_features, database_entries, db_feature_matrix, top_k=args.top_k)
    print_results_verbose(results, os.path.basename(args.query_image), args.top_k)

def run_batch_evaluation_mode(args):
    """Handles logic for batch evaluation, measuring spatial error of the top match."""
    print(f"\n--- Starting Batch Evaluation Mode ---")
    print(f"Querying every {args.batch_query_from_db}th image.")
    print(f"Exclusion radius around query: {args.batch_exclude_radius} images (by name).")
    print(f"Top-K results to consider for best match: {args.top_k}")
    if args.output_csv: print(f"Saving detailed results to: {args.output_csv}")

    full_db_entries, _ = load_and_filter_database(database_path=args.db_file)
    if not full_db_entries: print("Cannot run evaluation on an empty database."); return

    query_entries = full_db_entries[::args.batch_query_from_db]
    print(f"Selected {len(query_entries)} images to use as queries.")

    evaluation_results = []
    for i, query_entry in enumerate(query_entries):
        query_name = query_entry['base_name']
        try:
            numeric_part = int(query_name.split('.')[0])
            exclude_start, exclude_end = numeric_part - args.batch_exclude_radius, numeric_part + args.batch_exclude_radius
        except ValueError:
            exclude_start, exclude_end = None, None
            print(f"Warning: Non-numeric name '{query_name}', cannot apply exclusion radius.")
        
        search_db_entries, search_db_matrix = load_and_filter_database(
            all_database_entries=full_db_entries,
            exclude_range_start=exclude_start, exclude_range_end=exclude_end)
        
        results = retrieve_similar_images(query_entry['features'], search_db_entries, search_db_matrix, top_k=args.top_k)
        
        if results:
            top_match = results[0] # The best match is the first one in the sorted list
            distance = calculate_distance(query_entry, top_match)
            evaluation_results.append({
                'query_base_name': query_name,
                'exclude_radius': args.batch_exclude_radius,
                'top_match_base_name': top_match['base_name'],
                'distance_meters': distance,
                'similarity': top_match['similarity']
            })
            print(f"  [{i+1}/{len(query_entries)}] Query: {query_name} -> Top Match: {top_match['base_name']}, Distance: {distance:.2f}m")
        else:
            print(f"  [{i+1}/{len(query_entries)}] Query: {query_name} -> No matches found after exclusion.")

    # --- Final Summary Report ---
    print("\n--- Batch Evaluation Summary ---")
    if evaluation_results:
        distances = [res['distance_meters'] for res in evaluation_results]
        print(f"Evaluated {len(distances)} queries.")
        print(f"  Average Distance Error: {np.mean(distances):.2f}m")
        print(f"  Median Distance Error:  {np.median(distances):.2f}m")
        print(f"  Min Distance Error:     {np.min(distances):.2f}m")
        print(f"  Max Distance Error:     {np.max(distances):.2f}m")
        
        if args.output_csv:
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=evaluation_results[0].keys())
                writer.writeheader()
                writer.writerows(evaluation_results)
            print(f"\nSuccessfully saved detailed results to {args.output_csv}")
    else:
        print("No matches were found for any query, so no metrics could be calculated.")
    print("---------------------------------")

# --- Main Entry and Helpers ---

def print_results_verbose(results, query_name, top_k):
    """Helper function to format and print detailed retrieval results."""
    if results:
        print(f"\nTop {top_k} results for query '{query_name}':")
        for i, res in enumerate(results, 1):
            print(f"  {i}. Path: {res['path']}, Similarity: {res['similarity']:.4f}")
            print(f"     Coords: (E: {res['easting']:.2f}, N: {res['northing']:.2f}), Orientation: {res['orientation']}")
    else:
        print("No results found.")

def main():
    parser = argparse.ArgumentParser(description="Geo-localize an image using MegaLoc. Supports single query and batch evaluation modes.")
    parser.add_argument("query_image", type=str, nargs='?', default=None, help="Path to a single query image. Omit to use batch evaluation mode.")
    
    # --- Batch Evaluation Options ---
    parser.add_argument("--batch_query_from_db", type=int, metavar='N', help="Batch mode: Query every Nth image from the database.")
    parser.add_argument("--batch_exclude_radius", type=int, default=5, help="Batch mode: Exclude a radius of R images around the query's numeric name.")
    parser.add_argument("--output_csv", type=str, help="Batch mode: Path to save a CSV file with detailed evaluation results.")
    
    # --- General Options ---
    parser.add_argument("--top_k", type=int, default=TOP_K_RESULTS, help="Number of top results to return.")
    parser.add_argument("--db_file", type=str, default=DATABASE_FILE, help="Path to the precomputed features database.")
    
    # --- Database Management ---
    parser.add_argument("--create_new_db", action='store_true', help="Force creation of a new database.")
    parser.add_argument("--db_info", action='store_true', help="Show database info and exit.")
    
    args = parser.parse_args()
    
    is_batch_mode = args.query_image is None
    if is_batch_mode and not args.batch_query_from_db:
        parser.error("For batch mode (no query_image), you must specify '--batch_query_from_db N'.")

    if args.db_info:
        db_creator = MegaLocDatabaseCreator(database_output_file=args.db_file)
        info = db_creator.get_database_info()
        if info: print(f"\nDB Info for {args.db_file}: {info['total_entries']} entries.")
        else: print(f"No database found at {args.db_file}")
        return

    if not ensure_database_exists(args.db_file, args.create_new_db): return

    if is_batch_mode:
        run_batch_evaluation_mode(args)
    else:
        run_single_query_mode(args)

if __name__ == "__main__":
    main()