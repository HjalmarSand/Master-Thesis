import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import constants

from . import fourier_skyline_encoder


def load_fingerprints(filepath: Path) -> dict:
    """Loads fingerprints from a .npz file into a dictionary."""
    if not filepath.is_file():
        print(f"Error: Fingerprint file not found at {filepath}")
        return None
    return np.load(filepath)


def find_top_n_matches(
    query_key: str,
    query_vector: np.ndarray,
    database_vectors: dict,
    n: int = 5,
    num_windows: int = 4,
) -> list:
    """
    Finds the top N best matches by comparing fingerprints on a per-window basis.
    Matches are ranked by the average similarity across all windows.

    Args:
        query_key: The key of the query vector, to exclude from results.
        query_vector: The 1D numpy array of the fingerprint to match.
        database_vectors: A dictionary where keys are names and values are the fingerprints.
        n: The number of top matches to return.
        num_windows: The number of windows the fingerprint is composed of.

    Returns:
        A list of dictionaries, each containing the match key, average score, and window scores.
    """
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or num_windows <= 0:
        return []

    # Split the query vector into its window components
    query_windows = np.array_split(query_vector, num_windows)

    all_results = []
    for key, db_vector in database_vectors.items():
        if key == query_key:
            continue

        db_windows = np.array_split(db_vector, num_windows)

        # Ensure the vectors have the same window structure
        if len(query_windows) != len(db_windows):
            continue

        window_scores = []
        for q_win, db_win in zip(query_windows, db_windows):
            # Since vectors are pre-normalized, dot product is cosine similarity
            similarity = np.dot(q_win, db_win)
            window_scores.append(similarity)

        if window_scores:
            average_score = np.mean(window_scores)
            all_results.append(
                {
                    "match_key": key,
                    "avg_score": average_score,
                    "window_scores": window_scores,
                }
            )

    # Sort the list of dictionaries by average score in descending order
    all_results.sort(key=lambda item: item["avg_score"], reverse=True)

    return all_results[:n]


def run_self_matching(
    fingerprint_path: Path, num_samples: int = 10, top_n: int = 5, num_windows: int = 4
) -> list:
    """
    Performs a self-matching analysis on a single fingerprint file.
    """
    print("Loading fingerprints for self-matching...")
    fingerprints = load_fingerprints(fingerprint_path)
    if fingerprints is None:
        return None

    all_keys = list(fingerprints.keys())
    if len(all_keys) < 2:
        print("Not enough fingerprints in the file to perform a match.")
        return []

    np.random.seed(42)
    sample_keys = np.random.choice(
        all_keys, size=min(num_samples, len(all_keys)), replace=False
    )

    results = []
    print(
        f"Matching {len(sample_keys)} random samples against the database of {len(all_keys)} fingerprints..."
    )
    for key in tqdm(sample_keys, desc="Finding Top 5 Matches"):
        query_vector = fingerprints[key]
        top_matches = find_top_n_matches(
            key, query_vector, fingerprints, n=top_n, num_windows=num_windows
        )
        results.append({"query": key, "top_matches": top_matches})

    print("Self-matching complete.")
    return results


def run_cross_matching(
    query_fp_path: Path,
    dem_fp_path: Path,
    gps_csv_path: Path,
    num_samples: int = 20,
    top_n: int = 5,
    num_windows: int = constants.NUM_WINDOWS,
):
    """
    Matches query fingerprints (real images) against a DEM database using
    the robust window-averaging method.
    """
    # 1. Load Databases
    query_data = load_fingerprints(query_fp_path)
    dem_data = load_fingerprints(dem_fp_path)
    gps_df = pd.read_csv(gps_csv_path)

    gps_lookup = {
        row.filename: (row.easting, row.northing) for row in gps_df.itertuples()
    }
    query_keys = list(query_data.keys())

    # 3. Sample queries
    if len(query_keys) > num_samples:
        sample_indices = np.random.choice(len(query_keys), num_samples, replace=False)
        sample_keys = [query_keys[i] for i in sample_indices]
    else:
        sample_keys = query_keys

    # 4. Perform matching for each query
    all_results = []
    total_error = 0
    num_success = 0

    for query_key in tqdm(sample_keys, desc="Cross-Matching Queries"):
        query_vector = query_data[query_key]

        # Use the same robust matching function as the self-matching stage.
        # It compares window-by-window and averages the scores.
        top_matches = find_top_n_matches(
            query_key, query_vector, dem_data, n=top_n, num_windows=num_windows
        )

        query_num_str = re.search(r"(\d{4})", query_key)
        if not query_num_str:
            continue
        original_filename = f"{query_num_str.group(1)}.jpg"
        if original_filename not in gps_lookup:
            continue
        query_easting, query_northing = gps_lookup[original_filename]

        top_matches_info = []
        for i, match in enumerate(top_matches):
            match_key = match["match_key"]
            match_score = match["avg_score"]  # Use the average score

            parts = match_key.split("_")
            match_easting, match_northing = float(parts[2]), float(parts[3])
            error = np.sqrt(
                ((query_easting - match_easting) ** 2)
                + ((query_northing - match_northing) ** 2)
            )

            top_matches_info.append(
                {"match_key": match_key, "score": match_score, "error_m": error}
            )

            if i == 0:
                total_error += error
                num_success += 1

        all_results.append({"query": query_key, "top_matches": top_matches_info})

    mean_error = total_error / num_success if num_success > 0 else 0
    return all_results, mean_error


def find_yaw_offset(dem_mask: np.ndarray, real_mask: np.ndarray) -> float:
    """
    Calculates the rotational yaw offset between a real and DEM skyline.

    Args:
        dem_mask: The correctly oriented DEM skyline mask.
        real_mask: The real-world segmented panorama mask.

    Returns:
        The rotational offset in degrees needed to align the real mask with the DEM mask.
    """
    # 1. Extract the 1D skyline vectors from the masks
    dem_vector = fourier_skyline_encoder.extract_skyline_from_array(dem_mask)
    real_vector = fourier_skyline_encoder.extract_skyline_from_array(real_mask)

    if (
        dem_vector.size == 0
        or real_vector.size == 0
        or len(dem_vector) != len(real_vector)
    ):
        return 0.0

    best_offset = 0
    min_error = float("inf")

    # 2. Iterate through all possible pixel shifts (rotations)
    for shift in range(len(real_vector)):
        # Roll the real vector to simulate rotation
        shifted_real_vector = np.roll(real_vector, shift)

        # 3. Calculate the error (Mean Squared Error) between the two vectors
        error = np.mean((dem_vector - shifted_real_vector) ** 2)

        # 4. Keep track of the shift that results in the lowest error
        if error < min_error:
            min_error = error
            best_offset = shift

    # 5. Convert the best pixel offset to degrees
    image_width = len(real_vector)
    # Handle wrap-around: a large positive shift is a small negative rotation
    if best_offset > image_width / 2:
        best_offset = best_offset - image_width

    offset_degrees = (best_offset / image_width) * 360.0

    return offset_degrees
