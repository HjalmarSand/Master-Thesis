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


class CnnEncoder:
    """
    Encodes binary skyline images into feature vectors using a pre-trained CNN.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing CNN Encoder on device: {self.device}")

        # 1. Load a pre-trained model (MobileNetV2 is lightweight and effective)
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)

        # 2. Modify the model to remove the final classification layer.
        #    The 'features' part of MobileNetV2 is the convolutional base.
        self.feature_extractor = model.features.to(self.device)
        self.feature_extractor.eval()  # Set the model to evaluation mode

        # 3. Define the image transformations required by the pre-trained model.
        self.transform = transforms.Compose(
            [
                # TODO: The step below only works when we are working with 1280x960 aspect ratio for one window. Maybe we would like to create a function that does it best to resize maintaining aspect ratio and then crops in that
                # 1. Resize the smaller edge to 256 pixels, keeping the aspect ratio.
                transforms.Resize(256),
                # 2. Take a 224x224 crop from the center of the resized image.
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Pre-trained models expect 3 channels, so we duplicate the single channel.
                # The normalization values are standard for models trained on ImageNet.
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()  # Disable gradient calculations for efficiency
    def encode_image_window(self, image_window: Image.Image) -> np.ndarray:
        """
        Encodes a single image window into a feature vector.
        """
        # Apply the transformations and add a batch dimension
        img_tensor = self.transform(image_window).unsqueeze(0).to(self.device)

        # Pass the tensor through the feature extractor
        features = self.feature_extractor(img_tensor)

        # TODO: Maybe we want to make modifications here for different encodings and encoding sizes. We could for example pool down to (2, 2) instead and flatten if we want more details but keep robustness. Max pooling could perhaps be useful to experiment with also

        # Use adaptive average pooling to get a fixed-size feature vector
        # regardless of minor variations in feature map size.
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

        # Flatten the features and convert to a numpy array
        return pooled_features.squeeze().cpu().numpy()

    def get_windowed_fingerprint(
        self, image_path: Path, num_windows: int = 4
    ) -> np.ndarray:
        """
        Creates a spatially-aware fingerprint for a full panorama by encoding
        it in overlapping windows.
        """
        try:
            with Image.open(image_path).convert("L") as img:
                img_width, img_height = img.size
                window_width = img_width // num_windows

                # We'll use a small overlap to ensure features at the boundaries are captured
                overlap = window_width // 4
                step = window_width - overlap

                window_fingerprints = []
                for i in range(num_windows):
                    start = i * step
                    end = start + window_width

                    # Crop the window from the full panorama
                    window_pil = img.crop((start, 0, end, img_height))

                    # Encode the single window
                    window_vector = self.encode_image_window(window_pil)
                    window_fingerprints.append(window_vector)

                # Concatenate all window vectors into the final meta-fingerprint
                final_fingerprint = np.concatenate(window_fingerprints)

                # Normalize the final vector for stable cosine similarity comparisons
                norm = np.linalg.norm(final_fingerprint)
                return final_fingerprint / norm if norm > 0 else final_fingerprint

        except Exception as e:
            print(f"Warning: Could not process {image_path.name}. Error: {e}")
            return np.array([])


def create_cnn_fingerprints_for_directory(input_dir: Path, output_filepath: Path):
    """
    Main function to process a directory of panoramas and save their CNN fingerprints.
    """
    # Create the parent directory for the output file
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 1

    # Initialize the encoder once for the whole batch
    encoder = CnnEncoder()

    fingerprints = {}
    for image_path in tqdm(image_files, desc="Encoding with CNN"):
        fingerprint = encoder.get_windowed_fingerprint(image_path, num_windows=4)
        if fingerprint.size > 0:
            fingerprints[image_path.stem] = fingerprint

    if not fingerprints:
        print("No CNN fingerprints were generated.")
        return 1

    np.savez_compressed(output_filepath, **fingerprints)
    print(f"\nCNN fingerprinting complete. Data saved to '{output_filepath}'")
    return 0
