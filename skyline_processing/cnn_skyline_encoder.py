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
from AI.fine_tune_encoder import (
    EmbeddingNet,
    ValBinaryTransform,  # adjust import
)


class CnnEncoder:
    """
    Encodes binary skyline arrays into spatially-aware feature vectors.
    """

    def __init__(self, pool_target: tuple = (1, 1), model_path: Path = None):
        """
        Initializes the encoder.
        Args:
            pool_target (tuple): The grid size to pool features to.
            model_path (Path, optional): Path to a fine-tuned model state_dict.
                                        If None, uses default MobileNetV2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pool_target = pool_target
        print(f"Initializing CNN Encoder on device: {self.device}")

        if model_path and model_path.exists():
            print(f"Loading fine-tuned model from: {model_path}")
            model = EmbeddingNet(embedding_dim=128).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.feature_extractor = model.backbone.features
            # Use ValBinaryTransform → matches training preprocessing
            self.transform = ValBinaryTransform(target_h=224, w_shrink=2)
            self.uses_finetuned = True
        else:
            print("Fine-tuned model not found. Using default MobileNetV2.")
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
            self.feature_extractor = model.features.to(self.device)
            # Use simple resize pipeline → matches ImageNet training
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), antialias=True),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.uses_finetuned = False

        self.feature_extractor.eval()
        self.pool = torch.nn.functional.adaptive_avg_pool2d

    @torch.no_grad()
    def _encode_pil_window(self, image_window: Image.Image) -> np.ndarray:
        """Encodes a single PIL image window into a feature vector."""
        if self.uses_finetuned:
            img_t, _ = self.transform(
                image_window
            )  # ValBinaryTransform returns (tensor, profile)
        else:
            img_t = self.transform(image_window)  # Compose returns just tensor

        features = self.feature_extractor(img_t.unsqueeze(0).to(self.device))
        pooled_features = self.pool(features, self.pool_target)

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
    lazy_loaded_data, pool_target: tuple = (2, 2), model_path: Path = None
) -> dict:
    """
    Processes a large, lazy-loaded .npz file (DEM database).
    """
    encoder = CnnEncoder(pool_target=pool_target, model_path=model_path)
    fingerprints = {}

    for point_id in tqdm(lazy_loaded_data.files, desc="Encoding DEM with CNN"):
        mask_array = lazy_loaded_data[point_id]
        fingerprint = encoder.encode_mask_array(mask_array)
        if fingerprint.size > 0:
            fingerprints[point_id] = fingerprint

    return fingerprints


def create_cnn_fingerprints_for_directory(
    input_dir: Path,
    output_filepath: Path,
    pool_target: tuple = (2, 2),
    model_path: Path = None,
):
    """
    Processes a directory of real image panoramas.
    """
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    encoder = CnnEncoder(pool_target=pool_target, model_path=model_path)
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
