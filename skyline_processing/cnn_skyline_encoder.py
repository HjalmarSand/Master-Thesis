# We will be using a model that has been trained on detecting fundamental features like edges, corners and gradients.
# The segformer model gave us a good skyline segmentation in most panoramas, with some artifacts at times, and it has
# been trained on classifying specific features like trees buildings, sky and so on. Those features will not perform well
# when creating an encoding for a black and white binary mask. Perhaps it would work anyway, but we will begin with MobileNetV2

from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import constants


class CnnEncoder:
    """
    Encodes binary skyline arrays into spatially-aware feature vectors.
    """

    def __init__(self, pool_target: tuple = (1, 1)):
        """
        Initializes the encoder.
        Args:
            pool_target (tuple): The grid size to pool features to, e.g., (1, 1) for
                                global pooling or (2, 2) for a 4-chunk grid.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pool_target = pool_target
        print(
            f"Initializing CNN Encoder on device: {self.device} with {pool_target} pooling grid."
        )

        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        self.feature_extractor = model.features.to(self.device)
        self.feature_extractor.eval()

        self.pool = torch.nn.functional.adaptive_avg_pool2d

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def _encode_pil_window(self, image_window: Image.Image) -> np.ndarray:
        """Encodes a single PIL image window into a feature vector."""
        img_tensor = self.transform(image_window).unsqueeze(0).to(self.device)
        features = self.feature_extractor(img_tensor)

        # --- THIS IS THE KEY CHANGE ---
        # Pool to the configured grid size (e.g., 2x2) instead of just 1x1
        pooled_features = self.pool(features, self.pool_target)

        # Flatten the grid into a 1D vector and convert to numpy
        return pooled_features.flatten().cpu().numpy()

    def encode_mask_array(
        self, mask_array: np.ndarray, num_windows: int = constants.NUM_WINDOWS
    ) -> np.ndarray:
        """
        Primary method to create a fingerprint from an in-memory NumPy mask array.
        """
        if mask_array.ndim != 2 or mask_array.size == 0:
            return np.array([])

        try:
            # Convert NumPy array to PIL Image to use torchvision transforms
            img = Image.fromarray(mask_array).convert("L")
            img_width, img_height = img.size
            window_width = img_width // num_windows
            overlap = window_width // 4
            step = window_width - overlap

            window_fingerprints = []
            for i in range(num_windows):
                start = i * step
                end = start + window_width
                window_pil = img.crop((start, 0, end, img_height))
                window_vector = self._encode_pil_window(window_pil)
                window_fingerprints.append(window_vector)

            final_fingerprint = np.concatenate(window_fingerprints)
            norm = np.linalg.norm(final_fingerprint)
            return final_fingerprint / norm if norm > 0 else final_fingerprint
        except Exception as e:
            print(f"Warning: Could not process mask array. Error: {e}")
            return np.array([])


def create_cnn_fingerprints_from_lazy_loader(
    lazy_loaded_data, pool_target: tuple = (2, 2)
) -> dict:
    """
    NEW: Processes a large, lazy-loaded .npz file (DEM database).
    """
    encoder = CnnEncoder(pool_target=pool_target)
    fingerprints = {}

    for point_id in tqdm(lazy_loaded_data.files, desc="Encoding DEM with CNN"):
        mask_array = lazy_loaded_data[point_id]
        fingerprint = encoder.encode_mask_array(mask_array)
        if fingerprint.size > 0:
            fingerprints[point_id] = fingerprint

    return fingerprints


def create_cnn_fingerprints_for_directory(
    input_dir: Path, output_filepath: Path, pool_target: tuple = (2, 2)
):
    """
    Processes a directory of real image panoramas.
    """
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    encoder = CnnEncoder(pool_target=pool_target)
    fingerprints = {}
    for image_path in tqdm(image_files, desc="Encoding with CNN"):
        with Image.open(image_path).convert("L") as img:
            mask_array = np.array(img)
            fingerprint = encoder.encode_mask_array(mask_array)
            if fingerprint.size > 0:
                fingerprints[image_path.stem] = fingerprint

    np.savez_compressed(output_filepath, **fingerprints)
    print(f"\nCNN fingerprinting complete. Data saved to '{output_filepath}'")
    return 0
