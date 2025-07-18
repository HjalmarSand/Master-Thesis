from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import constants


def extract_skyline_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Extracts a 1D skyline vector from a 2D numpy mask array.
    This is the core skyline extraction logic.
    """
    if img_array.ndim != 2 or img_array.size == 0:
        return np.array([])

    img_height = img_array.shape[0]

    # Define a search window to make the search robust against noise
    search_start_y = (img_height // 2) + 20
    search_window = img_array[0:search_start_y, :]

    # Flip the window to search from the "bottom" (horizon) up
    flipped_window = np.flip(search_window, axis=0)

    # Find the index of the first white pixel (sky) in each column
    # This is the distance from the bottom of the *window*
    first_sky_pixel_idx = np.argmax(flipped_window > 128, axis=0)

    # Convert back to original image coordinates
    skyline_y = search_start_y - first_sky_pixel_idx

    # Handle columns that are all sky in the search window
    is_at_boundary = first_sky_pixel_idx == 0
    top_pixel_is_sky = img_array[0, :] > 128
    skyline_y[is_at_boundary & top_pixel_is_sky] = 0

    return skyline_y


def extract_skyline_from_image_path(image_path: Path) -> np.ndarray:
    """
    A wrapper function that opens an image from a path and extracts its skyline vector.
    """
    try:
        with Image.open(image_path).convert("L") as img:
            img_array = np.array(img)
            return extract_skyline_from_array(img_array)
    except Exception as e:
        print(f"Warning: Could not process {image_path.name}. Error: {e}")
        return np.array([])


def _calculate_fourier_magnitude(vector: np.ndarray) -> np.ndarray:
    """Helper to compute the normalized Fourier magnitude."""
    if vector.size < 2:
        return np.array([])
    fft_result = np.fft.fft(vector)
    magnitude = np.abs(fft_result)
    half_point = len(magnitude) // 2
    fingerprint = magnitude[1:half_point]  # Exclude DC component
    norm = np.linalg.norm(fingerprint)
    return fingerprint / norm if norm > 0 else fingerprint


def encode_single_mask(
    mask_array: np.ndarray, strategy: str, num_windows: int = constants.NUM_WINDOWS
) -> np.ndarray:
    """
    Encodes a single mask array into a Fourier fingerprint using the specified strategy.
    This is the primary, flexible encoding function.
    """
    skyline_vector = extract_skyline_from_array(mask_array)
    if skyline_vector.size == 0:
        return np.array([])

    # A power of 2 is often good for FFTs. This ensures all vectors are comparable.
    FIXED_SKYLINE_LENGTH = 4096
    if len(skyline_vector) != FIXED_SKYLINE_LENGTH:
        # Create x-axes for the original and new vector lengths
        original_x = np.linspace(0, 1, len(skyline_vector))
        new_x = np.linspace(0, 1, FIXED_SKYLINE_LENGTH)
        # Use linear interpolation to resample the vector to the fixed length
        skyline_vector = np.interp(new_x, original_x, skyline_vector)

    if strategy == "windowed":
        windows = np.array_split(skyline_vector, num_windows)
        all_fingerprints = [_calculate_fourier_magnitude(win) for win in windows]
        return np.concatenate(all_fingerprints)
    elif strategy == "global":
        return _calculate_fourier_magnitude(skyline_vector)
    else:
        return np.array([])


def create_fingerprints_from_lazy_loader(lazy_loaded_data, strategy: str) -> dict:
    """
    Processes a large, lazy-loaded .npz file and creates fingerprints.
    """
    fingerprints = {}
    # lazy_loaded_data.files gives the list of all point_ids
    for point_id in tqdm(lazy_loaded_data.files, desc=f"Encoding DEM ({strategy})"):
        # The mask is only loaded from disk when accessed here
        mask_array = lazy_loaded_data[point_id]
        fingerprint = encode_single_mask(mask_array, strategy=strategy)
        if fingerprint.size > 0:
            fingerprints[point_id] = fingerprint
    return fingerprints


def create_fingerprints_for_directory(
    input_dir: Path, output_filepath: Path, strategy: str = "windowed"
):
    """
    Processes all binary images in a directory (for real images).
    """
    if strategy not in ["global", "windowed"]:
        print(f"Error: Unknown strategy '{strategy}'.")
        return 1

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    fingerprints = {}
    for image_path in tqdm(image_files, desc=f"Fingerprinting ({strategy})"):
        # The logic is now simpler, just calling the main encoder
        with Image.open(image_path).convert("L") as img:
            mask_array = np.array(img)
            fingerprint = encode_single_mask(mask_array, strategy=strategy)
            if fingerprint.size > 0:
                fingerprints[image_path.stem] = fingerprint

    np.savez_compressed(output_filepath, **fingerprints)
    print(f"\nFingerprinting complete. Data saved to '{output_filepath}'")
    return 0
