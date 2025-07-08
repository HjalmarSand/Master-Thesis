from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_skyline_vector(image_path: Path) -> np.ndarray:
    """
    Extracts a 1D skyline vector from a binary skyline image.

    The function assumes the sky is white (255) and the ground/buildings
    are black (0). It finds the first non-sky pixel from the top of each
    column to represent the skyline's height.

    Args:
        image_path: The path to the binary image file.

    Returns:
        A 1D numpy array representing the skyline height for each column.
    """
    try:
        with Image.open(image_path).convert("L") as img:
            # Convert image to a numpy array
            img_array = np.array(img)

            # Find the index of the first non-white pixel (value < 128) in each column.
            # np.argmax on a boolean array returns the index of the first True value.
            skyline_vector = np.argmax(img_array < 128, axis=0)

            # If a column is all sky (all white), argmax returns 0. This should not happen though
            return skyline_vector
    except Exception as e:
        print(f"Warning: Could not process {image_path.name}. Error: {e}")
        return np.array([])


def calculate_fourier_fingerprint(skyline_vector: np.ndarray) -> np.ndarray:
    """
    Calculates a rotation-invariant fingerprint from a 1D skyline vector
    using the magnitude of its Fourier Transform.

    Args:
        skyline_vector: The 1D numpy array of skyline heights.

    Returns:
        A compact, rotation-invariant 1D numpy array fingerprint.
    """
    if skyline_vector.size == 0:
        return np.array([])

    # 1. Compute the Fast Fourier Transform
    fft_result = np.fft.fft(skyline_vector)

    # 2. Compute the magnitude spectrum (absolute value). This is the key
    #    step for rotation invariance.
    magnitude_spectrum = np.abs(fft_result)

    # 3. The spectrum is symmetric for real inputs, so we only need the first half
    half_point = len(magnitude_spectrum) // 2
    fingerprint = magnitude_spectrum[:half_point]

    # 4. Normalize the fingerprint to a unit vector (L2 norm). This makes
    #    comparisons more stable regardless of overall skyline height.
    norm = np.linalg.norm(fingerprint)
    if norm > 0:
        return fingerprint / norm
    else:
        return fingerprint


def create_fingerprints_for_directory(input_dir: Path, output_dir: Path):
    """
    Processes all binary images in a directory, creates a skyline fingerprint
    for each, and saves them all into a single compressed file.

    Args:
        input_dir: Directory containing the binary panorama images.
        output_dir: Directory where the fingerprint file will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    print(f"Found {len(image_files)} images to fingerprint in {input_dir}.")

    fingerprints = {}
    for image_path in tqdm(image_files, desc="Fingerprinting Skylines"):
        skyline_vector = extract_skyline_vector(image_path)
        if skyline_vector.size > 0:
            fingerprint = calculate_fourier_fingerprint(skyline_vector)
            # Use the image filename without extension as the key
            fingerprints[image_path.stem] = fingerprint

    if not fingerprints:
        print("No fingerprints were generated.")
        return 1

    # Save the entire dictionary of fingerprints to a single compressed .npz file
    output_file = output_dir / "skyline_fingerprints.npz"
    np.savez_compressed(output_file, **fingerprints)

    print(f"\n✅ Fingerprinting complete. Data saved to '{output_file}'")
    return 0
