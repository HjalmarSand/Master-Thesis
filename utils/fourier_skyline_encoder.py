# utils/skyline_fourier_encoder.py

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_skyline_vector(image_path: Path) -> np.ndarray:
    """
    Extracts a 1D skyline vector from a binary skyline image using a
    robust bottom-up search in a defined window.

    This method is designed to be robust against noise (black artifacts)
    in the upper sky portion of the image.
    """
    try:
        with Image.open(image_path).convert("L") as img:
            img_array = np.array(img)
            img_height = img_array.shape[0]

            # 1. Define the search window. Start 20px below the horizon.
            search_start_y = (img_height // 2) + 20
            search_window = img_array[0:search_start_y, :]

            # 2. Flip the window upside down to search from the bottom up.
            flipped_window = np.flip(search_window, axis=0)

            # 3. Find the index of the first WHITE pixel (sky) from the "bottom" up.
            first_sky_pixel_from_bottom_idx = np.argmax(flipped_window > 128, axis=0)

            # 4. Convert the index back to the original image's coordinate system.
            skyline_y = search_start_y - first_sky_pixel_from_bottom_idx

            # 5. Handle the edge case where a column is entirely sky within the search window.
            is_at_boundary = first_sky_pixel_from_bottom_idx == 0
            top_pixel_is_sky = img_array[0, :] > 128
            skyline_y[is_at_boundary & top_pixel_is_sky] = 0

            return skyline_y

    except Exception as e:
        print(f"Warning: Could not process {image_path.name}. Error: {e}")
        return np.array([])


def _calculate_fourier_magnitude(vector: np.ndarray) -> np.ndarray:
    """Helper function to compute the normalized Fourier magnitude for a vector."""
    if vector.size == 0:
        return np.array([])

    fft_result = np.fft.fft(vector)
    magnitude_spectrum = np.abs(fft_result)
    half_point = len(magnitude_spectrum) // 2
    fingerprint = magnitude_spectrum[:half_point]
    norm = np.linalg.norm(fingerprint)
    return fingerprint / norm if norm > 0 else fingerprint


def calculate_global_fourier_fingerprint(skyline_vector: np.ndarray) -> np.ndarray:
    """Calculates a single, rotation-invariant fingerprint for the entire skyline."""
    return _calculate_fourier_magnitude(skyline_vector)


def calculate_windowed_fourier_fingerprint(
    skyline_vector: np.ndarray, num_windows: int = 4
) -> np.ndarray:
    """Calculates a spatially-aware fingerprint by dividing the skyline into windows."""
    if skyline_vector.size == 0:
        return np.array([])
    skyline_windows = np.array_split(skyline_vector, num_windows)
    all_fingerprints = [
        _calculate_fourier_magnitude(window) for window in skyline_windows
    ]
    return np.concatenate(all_fingerprints)


def create_fingerprints_for_directory(
    input_dir: Path, output_filepath: Path, strategy: str = "windowed"
):
    """
    Processes all binary images in a directory and saves their fingerprints to a specific file.

    Args:
        input_dir: Directory containing the binary panorama images.
        output_filepath: The full path for the output .npz file.
        strategy: The fingerprinting method to use ('global' or 'windowed').
    """
    if strategy not in ["global", "windowed"]:
        print(
            f"Error: Unknown strategy '{strategy}'. Please use 'global' or 'windowed'."
        )
        return 1

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    print(
        f"Found {len(image_files)} images. Generating fingerprints using '{strategy}' strategy."
    )

    fingerprints = {}
    for image_path in tqdm(image_files, desc=f"Fingerprinting ({strategy})"):
        skyline_vector = extract_skyline_vector(image_path)
        if skyline_vector.size > 0:
            if strategy == "windowed":
                fingerprint = calculate_windowed_fourier_fingerprint(
                    skyline_vector, num_windows=4
                )
            else:  # global
                fingerprint = calculate_global_fourier_fingerprint(skyline_vector)

            fingerprints[image_path.stem] = fingerprint

    if not fingerprints:
        print("No fingerprints were generated.")
        return 1

    np.savez_compressed(output_filepath, **fingerprints)

    print(f"\n✅ Fingerprinting complete. Data saved to '{output_filepath}'")
    return 0
