import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sympy import false
from tqdm import tqdm

try:
    current_script_dir = Path(__file__).resolve().parent
    parent_dir = current_script_dir.parent
    sys.path.append(str(parent_dir))
    import constants

    DATASETS_ROOT_DIR = Path(constants.IMAGES_DIR)
except (ImportError, NameError):
    traceback.print_exc()


def skyline_profile_from_image(img: Image.Image, normalize: bool = True) -> np.ndarray:
    """Converts a 2D binary PIL Image into a 1D skyline profile vector."""
    arr = np.array(img.convert("L"))
    H, _ = arr.shape
    sky = (arr > 128).astype(int)
    profile = H - np.argmax(np.flipud(sky), axis=0)
    all_sky_cols = sky.sum(axis=0) == H
    profile[all_sky_cols] = 0
    if normalize:
        profile = profile.astype(float) - profile.mean()
    return profile


def compare_skyline_profiles(profile1: np.ndarray, profile2: np.ndarray) -> float:
    """
    Compares two skyline profiles, finding the best alignment via circular shifting.
    Returns only the minimum error found.
    """
    width = len(profile1)
    min_error = float("inf")

    # Pre-calculate errors for all shifts using fast Fourier transform for correlation
    # The result of correlation is equivalent to calculating squared error at each lag.
    # We use MAE, so we stick to the loop, which is clear and fast enough.
    for shift in range(width):
        rolled_profile2 = np.roll(profile2, shift)
        error = np.mean(np.abs(profile1 - rolled_profile2))
        if error < min_error:
            min_error = error
    return min_error


def load_data_paths(
    dataset_dirs: list[Path], val_split: float = 0.1, random_state: int = 42
) -> tuple[dict, dict]:
    """
    Loads and splits data paths.
    Returns two dictionaries: one for the validation set (queries) and
    one for the full database (candidates).
    """
    rng = np.random.default_rng(random_state)
    val_set = {}
    db_set = {}
    label_counter = 0

    print(f"Scanning {len(dataset_dirs)} dataset directories...")
    for ds_dir in dataset_dirs:
        real_dir = ds_dir / "binary_panorama_aligned"
        dem_dir = ds_dir / "dem_skylines_paired"
        gps_csv = ds_dir / "gps_interpolated_with_orientation.csv"

        if not (real_dir.is_dir() and dem_dir.is_dir() and gps_csv.is_file()):
            print(f"Skipping {ds_dir.name}: Required data not found.")
            continue

        # Use IoU/Yaw filtering here?

        gps_df = pd.read_csv(gps_csv)
        fname_col = [c for c in gps_df.columns if c.lower() in ["filename", "file"]][0]

        ids = []
        id_to_paths = {}
        for _, row in gps_df.iterrows():
            image_number = Path(row[fname_col]).stem
            real_path = real_dir / f"binary_panorama_{image_number}.png"
            dem_path = dem_dir / f"{image_number}.npy"
            if real_path.exists() and dem_path.exists():
                ids.append(image_number)
                id_to_paths[image_number] = (real_path, dem_path)

        # Train/Val split per dataset to ensure fair evaluation
        ids = np.array(sorted(ids))
        n = len(ids)
        n_val = max(1, int(round(n * val_split)))
        perm = rng.permutation(n)
        val_ids = ids[perm[:n_val]]

        for img_id in ids:
            real_path, dem_path = id_to_paths[img_id]
            current_label = label_counter

            db_set[current_label] = {"dem_path": dem_path, "real_path": real_path}

            if img_id in val_ids:
                val_set[current_label] = {"real_path": real_path}

            label_counter += 1

    print(
        f"Loaded {len(val_set)} validation queries and {len(db_set)} database entries."
    )
    return val_set, db_set


def compute_recall_at_k(
    query_labels: list, ranked_results: list[list], K_values: tuple
) -> dict:
    """
    Calculates Recall@K metrics.

    Args:
        query_labels: The ground truth label for each query.
        ranked_results: A list where each item is a sorted list of database labels.
        K_values: A tuple of K's to evaluate (e.g., (1, 5, 10)).
    """
    recalls = {k: 0.0 for k in K_values}
    num_queries = len(query_labels)

    for i in range(num_queries):
        true_label = query_labels[i]
        top_k_preds = ranked_results[i]
        for k in K_values:
            if true_label in top_k_preds[:k]:
                recalls[k] += 1

    for k in K_values:
        recalls[k] /= num_queries

    return recalls


def run_baseline_evaluation():
    """Main function to run the entire evaluation process."""
    # Standardize image dimensions for fair comparison
    IMG_WIDTH, IMG_HEIGHT = 2048, 200

    dataset_dirs = sorted([d for d in DATASETS_ROOT_DIR.iterdir() if d.is_dir()])
    if not dataset_dirs:
        print(f"No dataset directories found in {DATASETS_ROOT_DIR}.")
        return

    val_queries, db_items = load_data_paths(dataset_dirs)

    print("\nEncoding all DEM images in the database into profiles...")
    db_profiles = {}
    for label, item in tqdm(db_items.items(), desc="Encoding DB"):
        dem_path = item["dem_path"]
        dem_arr = np.load(dem_path)  # Assuming it's a .npy file
        # Assuming the .npy is already a 2D binary mask [0, 255]
        img = Image.fromarray(dem_arr.astype(np.uint8))
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
        db_profiles[label] = skyline_profile_from_image(img, normalize=True)

    print("\nRunning evaluation: comparing validation queries against the database...")
    query_labels = []
    ranked_results = []

    for true_label, query_item in tqdm(val_queries.items(), desc="Evaluating Queries"):
        query_labels.append(true_label)

        real_path = query_item["real_path"]
        real_img = Image.open(real_path).resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
        query_profile = skyline_profile_from_image(real_img, normalize=True)

        distances = []
        for db_label, db_profile in db_profiles.items():
            error = compare_skyline_profiles(query_profile, db_profile)
            distances.append((error, db_label))

        # Rank database items by distance
        distances.sort(key=lambda x: x[0])
        ranked_db_labels = [label for error, label in distances]
        ranked_results.append(ranked_db_labels)

    print("\nCalculating final metrics...")
    K_values = (1, 5, 10, 20)
    recalls = compute_recall_at_k(query_labels, ranked_results, K_values)

    print("\n--- Baseline Evaluation Results (Direct Profile Matching) ---")
    for k, r in recalls.items():
        print(f"Recall@{k:<2}: {r:.4f} ({r * 100:.2f}%)")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    run_baseline_evaluation()
