from pathlib import Path

import numpy as np
from tqdm import tqdm


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
