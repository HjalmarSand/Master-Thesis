# main_retrieval.py (with MegaLoc and exclusion)
import os
import pickle
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

DATABASE_FILE = "Hjalmar/image_database_megaloc.pkl"
TOP_K_RESULTS = 5
MEGALOC_INPUT_IMAGE_SIZE = (480, 640)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_model_and_transforms():
    print("Initializing MegaLoc model...")
    try:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
    except Exception as e:
        print(f"Error loading MegaLoc model from torch.hub: {e}"); exit()
    
    model.eval()
    model.to(device)
    preprocess = T.Compose([
        T.Resize(MEGALOC_INPUT_IMAGE_SIZE), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("MegaLoc model and transforms initialized.")
    return model, preprocess

def extract_features(image_path, model, preprocess_fn):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess_fn(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        with torch.no_grad():
            features = model(batch_t)
        features_flat = features.squeeze().cpu().numpy()
        return features_flat
    except FileNotFoundError:
        print(f"Error: Query image not found at {image_path}"); return None
    except Exception as e:
        print(f"Error processing query image {image_path}: {e}"); return None

def load_and_filter_database(database_path, exclude_list=None, exclude_range_start=None, exclude_range_end=None):
    """Loads the database and filters entries based on exclusion criteria."""
    try:
        with open(database_path, 'rb') as f:
            all_database_entries = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Database file not found at {database_path}. Run create_database.py first."); return None, None
    except Exception as e:
        print(f"Error loading database: {e}"); return None, None

    if not all_database_entries:
        print(f"Warning: Loaded database from {database_path} is empty."); return None, None

    filtered_entries = []
    if exclude_list is None:
        exclude_list = []
    
    # Normalize exclude_list to ensure ".jpg" suffix for base names if not present
    normalized_exclude_list = []
    for item in exclude_list:
        if not item.endswith(".jpg"):
            normalized_exclude_list.append(item + ".jpg")
        else:
            normalized_exclude_list.append(item)

    for entry in all_database_entries:
        base_name = entry["base_name"] 
        
        # Check direct exclusion list
        if base_name in normalized_exclude_list:
            print(f"Excluding {base_name} (from direct list)")
            continue

        # Check range exclusion
        if exclude_range_start and exclude_range_end:
            # Assuming base_name is "XXXX.jpg" where XXXX is numeric
            try:
                numeric_part_str = base_name.split('.')[0]
                if numeric_part_str.isdigit(): 
                    numeric_part = int(numeric_part_str)
                    if exclude_range_start <= numeric_part <= exclude_range_end:
                        print(f"Excluding {base_name} (from range {exclude_range_start}-{exclude_range_end})")
                        continue
                else:
                    pass 
            except ValueError:
                pass 
        
        filtered_entries.append(entry)

    if not filtered_entries:
        print(f"Warning: Database is empty after applying exclusions."); return None, None
        
    db_feature_matrix = np.array([item["features"] for item in filtered_entries])
    return filtered_entries, db_feature_matrix


def retrieve_similar_images(query_features, db_entries, db_feature_matrix, top_k=5):
    if query_features is None: return []
    if db_feature_matrix is None or db_feature_matrix.shape[0] == 0:
        print("Cannot retrieve, filtered database is empty.")
        return []
        
    query_features_2d = query_features.reshape(1, -1)
    similarities = cosine_similarity(query_features_2d, db_feature_matrix).flatten()
    top_k_indices = np.argsort(-similarities)[:top_k]
    results = []
    for i in top_k_indices:
        results.append({
            "path": db_entries[i]["path"], "base_name": db_entries[i]["base_name"],
            "easting": db_entries[i]["easting"], "northing": db_entries[i]["northing"],
            "orientation": db_entries[i]["orientation"], "similarity": similarities[i]
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Geo-localize an image using MegaLoc features with optional DB exclusion.")
    parser.add_argument("query_image", type=str, help="Path to the query image.")
    parser.add_argument("--top_k", type=int, default=TOP_K_RESULTS, help="Number of top results.")
    parser.add_argument("--db_file", type=str, default=DATABASE_FILE, help="Path to precomputed DB.")
    parser.add_argument("--exclude", nargs='*', help="List of base filenames to exclude (e.g., 0801 0802). '.jpg' is optional.")
    parser.add_argument("--exclude_range", type=str, help="Range of numeric base filenames to exclude (e.g., '0801-0901').")
    
    args = parser.parse_args()

    exclude_range_start, exclude_range_end = None, None
    if args.exclude_range:
        try:
            start_str, end_str = args.exclude_range.split('-')
            exclude_range_start = int(start_str)
            exclude_range_end = int(end_str)
            if exclude_range_start > exclude_range_end:
                print("Warning: exclude_range start is greater than end. No range exclusion will be applied.")
                exclude_range_start, exclude_range_end = None, None
        except ValueError:
            print(f"Warning: Invalid format for --exclude_range '{args.exclude_range}'. Expected 'START-END' (e.g., '0801-0901').")

    database_entries, db_feature_matrix = load_and_filter_database(
        args.db_file, 
        exclude_list=args.exclude, 
        exclude_range_start=exclude_range_start, 
        exclude_range_end=exclude_range_end
    )
    
    if database_entries is None or db_feature_matrix is None or db_feature_matrix.shape[0] == 0:
        print("Database is empty or became empty after filtering. Cannot proceed.")
        return
        
    print(f"Loaded and filtered MegaLoc DB. Using {len(database_entries)} entries from {args.db_file}")

    model, preprocess_fn = initialize_model_and_transforms()

    print(f"\nProcessing query image: {args.query_image}")
    query_features = extract_features(args.query_image, model, preprocess_fn)
    if query_features is None: return

    results = retrieve_similar_images(query_features, database_entries, db_feature_matrix, top_k=args.top_k)

    if results:
        print(f"\nTop {args.top_k} MegaLoc results for query '{os.path.basename(args.query_image)}' (from filtered DB):")
        for res in results:
            print(
                f"  Path: {res['path']}, Similarity: {res['similarity']:.4f}, "
                f"Easting: {res['easting']:.2f}, Northing: {res['northing']:.2f}, "
                f"Orientation: {res['orientation']}"
            )
    else:
        print("No results found (possibly due to empty filtered database or no matches).")


if __name__ == "__main__":
    main()