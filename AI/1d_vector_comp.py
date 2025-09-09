import sys
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import medfilt
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


def compare_skyline_profiles(
    profile1: np.ndarray, profile2: np.ndarray, search_range_deg: int
) -> float:
    """
    Compares two skyline profiles, finding the best alignment by shifting
    within a limited angular range. Returns only the minimum error found.
    """
    width = len(profile1)
    min_error = float("inf")

    pixels_per_degree = width / 360.0
    shift_radius = int(round(search_range_deg * pixels_per_degree))

    for shift in range(-shift_radius, shift_radius + 1):
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


def dem_vector_to_mask(
    elevations: np.ndarray,
    output_width: int,
    output_height: int,
    v_fov_deg: float = 60.0,
    median_kernel: int = 19,
) -> np.ndarray:
    e = np.asarray(elevations).squeeze()
    if e.ndim != 1:
        raise ValueError(f"DEM vector must be 1D after squeeze, got shape {e.shape}")

    num_bins = len(e)
    angles = np.linspace(0, 360, num_bins, endpoint=False)

    order = np.argsort(angles)
    a_sorted = angles[order]
    e_sorted = e[order]
    a_wrap = np.concatenate(([a_sorted[-1] - 360], a_sorted, [a_sorted[0] + 360]))
    e_wrap = np.concatenate(([e_sorted[-1]], e_sorted, [e_sorted[0]]))

    out_az = np.linspace(0, 360, output_width, endpoint=False)
    e_interp = np.interp(out_az, a_wrap, e_wrap, period=360)

    if median_kernel and median_kernel > 1 and median_kernel % 2 == 1:
        e_interp = medfilt(e_interp, kernel_size=median_kernel)

    top = v_fov_deg / 2.0
    rows = np.clip(
        np.round(((top - e_interp) / v_fov_deg) * output_height), 0, output_height
    ).astype(int)

    mask_sky_white = np.zeros((output_height, output_width), dtype=np.uint8)
    for x, y in enumerate(rows):
        mask_sky_white[0:y, x] = 255

    return mask_sky_white


def compute_recall_at_k(
    query_labels: list, ranked_results: list[list], K_values: tuple
) -> dict:
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
    """Main function using the new mask-generation workflow."""
    TARGET_WIDTH = 5120
    TARGET_HEIGHT = 960
    SEARCH_RANGE_DEG = 15
    DEM_V_FOV_DEG = 60.0

    dataset_dirs = sorted([d for d in DATASETS_ROOT_DIR.iterdir() if d.is_dir()])
    val_queries, db_items = load_data_paths(dataset_dirs)

    print("\nGenerating all DEM masks from the database...")
    db_profiles = {}
    for label, item in tqdm(db_items.items(), desc="Encoding DB"):
        dem_vector = np.load(item["dem_path"])
        dem_mask = dem_vector_to_mask(
            dem_vector,
            output_width=TARGET_WIDTH,
            output_height=TARGET_HEIGHT,
            v_fov_deg=DEM_V_FOV_DEG,
        )
        dem_profile = skyline_profile_from_image(Image.fromarray(dem_mask))
        db_profiles[label] = dem_profile - dem_profile.mean()

    print("\nRunning evaluation: comparing validation queries against the database...")
    query_labels = []
    ranked_results = []
    plot_generated = False  # Flag to ensure we only plot once

    for true_label, query_item in tqdm(val_queries.items(), desc="Evaluating Queries"):
        query_labels.append(true_label)

        real_path = query_item["real_path"]
        real_img = Image.open(real_path).resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST
        )
        query_profile = skyline_profile_from_image(real_img)
        query_profile = query_profile - query_profile.mean()

        distances = []
        for db_label, db_profile in db_profiles.items():
            error = compare_skyline_profiles(
                query_profile, db_profile, SEARCH_RANGE_DEG
            )
            distances.append((error, db_label))

        distances.sort(key=lambda x: x[0])
        ranked_db_labels = [label for _, label in distances]
        ranked_results.append(ranked_db_labels)

        if not plot_generated and ranked_db_labels[0] == true_label:
            plot_generated = True  # Set flag so we don't plot again
            print(f"\nFound a perfect match (Rank 1)! Label: {true_label}")
            print("Generating visual inspection plot... Close the plot to continue.")

            correct_dem_path = db_items[true_label]["dem_path"]
            dem_vector = np.load(correct_dem_path)
            correct_dem_mask = dem_vector_to_mask(
                dem_vector,
                output_width=TARGET_WIDTH,
                output_height=TARGET_HEIGHT,
                v_fov_deg=DEM_V_FOV_DEG,
            )

            correct_db_profile = db_profiles[true_label]

            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            fig.suptitle(
                f"Visual Inspection of Correct Match: Label {true_label}", fontsize=16
            )

            axes[0].set_title("Query: Real Image")
            axes[0].imshow(real_img, cmap="gray")
            axes[0].axis("off")

            axes[1].set_title("Match: Synthesized DEM Mask")
            axes[1].imshow(correct_dem_mask, cmap="gray")
            axes[1].axis("off")

            axes[2].set_title("Normalized Skyline Profile Comparison")
            axes[2].plot(query_profile, label="Query Profile (Real)", alpha=0.8)
            axes[2].plot(
                correct_db_profile, label="DB Profile (DEM)", alpha=0.8, linestyle="--"
            )
            axes[2].legend()
            axes[2].grid(True)
            axes[2].set_xlabel("Image Width (pixels)")
            axes[2].set_ylabel("Normalized Height")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    print("\nCalculating final metrics...")
    K_values = (1, 5, 10, 20)
    recalls = compute_recall_at_k(query_labels, ranked_results, K_values)

    print(f"\n--- Evaluation Results (DEM Mask Generation Workflow) ---")
    print(f"Search Range: +/- {SEARCH_RANGE_DEG} degrees, V_FOV: {DEM_V_FOV_DEG} deg")
    for k, r in recalls.items():
        print(f"Recall@{k:<2}: {r:.4f} ({r * 100:.2f}%)")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    run_baseline_evaluation()
