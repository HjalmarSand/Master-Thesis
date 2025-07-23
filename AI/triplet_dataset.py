import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class SkylineTripletDataset(Dataset):
    """
    Creates triplets of (anchor, positive, negative) images for metric learning.
    - Anchor: A real panoramic skyline mask.
    - Positive: The corresponding simulated DEM skyline mask for the same location.
    - Negative: A real panoramic skyline mask from a distant location.
    """

    def __init__(self, dataset_dir: Path, transform=None, margin_images: int = 50):
        """
        Args:
            dataset_dir (Path): Path to the processed dataset folder (e.g., 'images/dataset1').
            transform (callable, optional): Optional transform to be applied on a sample.
            margin_images (int): The minimum index separation for a negative sample.
        """
        self.transform = transform
        self.margin_images = margin_images

        real_skyline_dir = dataset_dir / "binary_panorama_aligned"
        dem_skyline_dir = dataset_dir / "dem_skylines_paired"
        gps_path = dataset_dir / "gps_interpolated_with_orientation.csv"

        if not all(
            [real_skyline_dir.is_dir(), dem_skyline_dir.is_dir(), gps_path.is_file()]
        ):
            raise FileNotFoundError(
                f"Required data not found in {dataset_dir}. Please run preceding pipeline stages."
            )

        gps_df = pd.read_csv(gps_path)

        self.anchor_paths = []
        self.positive_paths = []

        print("Loading and verifying triplet pairs...")
        for _, row in gps_df.iterrows():
            image_number = Path(row.filename).stem
            anchor_path = real_skyline_dir / f"binary_panorama_{image_number}.png"
            positive_path = dem_skyline_dir / f"{image_number}.npy"

            if anchor_path.exists() and positive_path.exists():
                self.anchor_paths.append(anchor_path)
                self.positive_paths.append(positive_path)

        if not self.anchor_paths:
            raise ValueError(f"No valid anchor/positive pairs found in {dataset_dir}")

        self.num_samples = len(self.anchor_paths)
        print(f"Found {self.num_samples} valid pairs.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 1. Load the raw PIL Images
        anchor_path = self.anchor_paths[index]
        anchor_img = Image.open(anchor_path).convert("L")

        positive_path = self.positive_paths[index]
        positive_array = np.load(positive_path)
        positive_img = Image.fromarray(positive_array.astype(np.uint8)).convert("L")

        while True:
            negative_index = random.randint(0, self.num_samples - 1)
            if abs(index - negative_index) > self.margin_images:
                break
        negative_path = self.anchor_paths[negative_index]
        negative_img = Image.open(negative_path).convert("L")

        # 2. Apply transformations
        if self.transform:
            # The first transform in our pipeline is RandomResizedCrop
            random_cropper = self.transform.transforms[0]

            # Get parameters for a random crop ONCE
            crop_params = random_cropper.get_params(
                anchor_img, random_cropper.scale, random_cropper.ratio
            )
            # Apply the SAME crop
            anchor_cropped = F.crop(anchor_img, *crop_params)
            positive_cropped = F.crop(positive_img, *crop_params)
            # Apply the SAME resize to get to the final size
            target_size = [random_cropper.size[0], random_cropper.size[1]]
            anchor_resized = F.resize(anchor_cropped, target_size, antialias=True)
            positive_resized = F.resize(positive_cropped, target_size, antialias=True)

            # Handle Negative Independently
            # Apply the full random crop + resize transform directly
            negative_resized = random_cropper(negative_img)

            # Apply the rest of the transforms to all three images
            # We start from the 1st index since the 0th (RandomResizedCrop) is done.
            final_anchor = anchor_resized
            final_positive = positive_resized
            final_negative = negative_resized

            for t in self.transform.transforms[1:]:
                final_anchor = t(final_anchor)
                final_positive = t(final_positive)
                final_negative = t(final_negative)

            return final_anchor, final_positive, final_negative

        # Fallback if no transform is provided
        return anchor_img, positive_img, negative_img
