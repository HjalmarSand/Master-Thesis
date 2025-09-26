from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import constants

SKY_THRESHOLD = 128  # sky = [128..255], ground = [0..127]
FIXED_SKYLINE_LENGTH = 4096  # power-of-two for FFTs


def extract_skyline_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Returns rows: shape (W,), row index (0=top) of first ground pixel per col.
    If a column is all sky, returns H (image height) at that column.
    """
    if img_array.ndim != 2 or img_array.size == 0:
        return np.array([], dtype=np.int32)

    arr = img_array.astype(np.uint8, copy=False)
    H, W = arr.shape

    sky = arr >= SKY_THRESHOLD
    ground = ~sky

    has_ground = ground.any(axis=0)  # (W,)
    first_ground = ground.argmax(axis=0)  # (W,), 0 if all False
    rows = np.where(has_ground, first_ground, H)  # (W,)
    return rows.astype(np.int32)


def extract_skyline_from_image_path(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path).convert("L") as img:
            img_array = np.array(img, dtype=np.uint8)
        return extract_skyline_from_array(img_array)
    except Exception as e:
        print(f"Warning: Could not process {image_path.name}. Error: {e}")
        return np.array([], dtype=np.int32)


def _calculate_fourier_magnitude(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalized rFFT magnitude, DC excluded.
    Input vector should be 1D float (e.g., normalized 0..1).
    """
    if vector.size < 2:
        return np.array([], dtype=np.float32)

    # real FFT -> length N//2+1 (bins 0..Nyquist)
    spec = np.fft.rfft(vector)
    mag = np.abs(spec).astype(np.float32)

    # drop DC component (index 0); keep up to Nyquist
    fp = mag[1:]
    norm = np.linalg.norm(fp)
    return (fp / norm).astype(np.float32) if norm > 0 else fp.astype(np.float32)


def encode_single_mask(
    mask_array: np.ndarray, strategy: str, num_windows: int = constants.NUM_WINDOWS
) -> np.ndarray:
    """
    Encodes a single mask array into a Fourier fingerprint using the specified strategy.
    strategy: "global" | "windowed"
    """
    skyline_rows = extract_skyline_from_array(mask_array)
    if skyline_rows.size == 0:
        return np.array([], dtype=np.float32)

    H = mask_array.shape[0]
    # normalize heights to [0,1] so amplitudes are comparable across images
    skyline = skyline_rows.astype(np.float32) / float(H)

    # resample horizontally to a fixed length so all FFTs are comparable
    if skyline.shape[0] != FIXED_SKYLINE_LENGTH:
        x_old = np.linspace(0.0, 1.0, skyline.shape[0], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, FIXED_SKYLINE_LENGTH, dtype=np.float32)
        skyline = np.interp(x_new, x_old, skyline).astype(np.float32)

    if strategy == "windowed":
        # ensure clean splits
        if FIXED_SKYLINE_LENGTH % num_windows != 0:
            raise ValueError(
                f"FIXED_SKYLINE_LENGTH ({FIXED_SKYLINE_LENGTH}) must be divisible by "
                f"num_windows ({num_windows})."
            )
        windows = np.array_split(skyline, num_windows)
        fps = [_calculate_fourier_magnitude(win) for win in windows]
        return np.concatenate(fps).astype(np.float32)

    elif strategy == "global":
        return _calculate_fourier_magnitude(skyline)

    else:
        return np.array([], dtype=np.float32)


def create_fingerprints_from_lazy_loader(lazy_loaded_data, strategy: str) -> dict:
    """
    Processes a large, lazy-loaded .npz file and creates fingerprints.
    Returns a dict: {point_id: np.ndarray[f32]}.
    """
    fingerprints = {}
    for point_id in tqdm(lazy_loaded_data.files, desc=f"Encoding DEM ({strategy})"):
        mask_array = lazy_loaded_data[point_id]
        fp = encode_single_mask(mask_array, strategy=strategy)
        if fp.size > 0:
            fingerprints[point_id] = fp
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
        with Image.open(image_path).convert("L") as img:
            mask_array = np.array(img, dtype=np.uint8)
        fp = encode_single_mask(mask_array, strategy=strategy)
        if fp.size > 0:
            fingerprints[image_path.stem] = fp

    np.savez_compressed(output_filepath, **fingerprints)
    print(f"\nFingerprinting complete. Data saved to '{output_filepath}'")
    return 0
