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

        # Define paths to the required data directories
        real_skyline_dir = dataset_dir / "binary_panorama_aligned"
        dem_skyline_dir = dataset_dir / "dem_skylines_paired"
        gps_path = dataset_dir / "gps_interpolated_with_orientation.csv"

        if not all(
            [real_skyline_dir.is_dir(), dem_skyline_dir.is_dir(), gps_path.is_file()]
        ):
            raise FileNotFoundError(
                f"Required data not found in {dataset_dir}. Please run preceding pipeline stages."
            )

        # Load GPS data to find corresponding filenames
        gps_df = pd.read_csv(gps_path)

        self.anchor_paths = []
        self.positive_paths = []

        print("Loading and verifying triplet pairs...")
        # Create a list of valid anchor/positive pairs
        for _, row in gps_df.iterrows():
            image_number = Path(row.filename).stem

            anchor_path = (
                real_skyline_dir / f"binary_panorama_aligned_{image_number}.png"
            )
            positive_path = dem_skyline_dir / f"{image_number}.npy"

            # Ensure both files for the pair actually exist before adding them
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
            # Get parameters for a random crop ONCE
            # We use the anchor image to get the parameters
            crop_params = self.transform.transforms[0].get_params(
                anchor_img,
                self.transform.transforms[0].scale,
                self.transform.transforms[0].ratio,
            )

            # Apply the SAME crop to anchor and positive
            anchor_img = F.crop(anchor_img, *crop_params)
            positive_img = F.crop(positive_img, *crop_params)

            # Apply a DIFFERENT random crop to the negative
            neg_crop_params = self.transform.transforms[0].get_params(
                negative_img,
                self.transform.transforms[0].scale,
                self.transform.transforms[0].ratio,
            )
            negative_img = F.crop(negative_img, *neg_crop_params)

            # Note: We skip the first transform (RandomResizedCrop) which we did manually
            for t in self.transform.transforms[1:]:
                anchor_img = t(anchor_img)
                positive_img = t(positive_img)
                negative_img = t(negative_img)

        return anchor_img, positive_img, negative_img
