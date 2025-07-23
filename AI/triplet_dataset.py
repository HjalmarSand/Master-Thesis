import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class SkylineTripletDataset(Dataset):
    """
    Creates triplets of (anchor, positive, negative) images for metric learning.
    Now includes functionality to split data into training and validation sets.
    """

    def __init__(
        self,
        dataset_dir: Path,
        transform=None,
        margin_images: int = 100,
        mode: str = "train",
        val_split: float = 0.1,
        random_state: int = 42,
    ):
        self.transform = transform
        self.margin_images = margin_images
        self.mode = mode

        real_skyline_dir = dataset_dir / "binary_panorama_aligned"
        dem_skyline_dir = dataset_dir / "dem_skylines_paired"
        gps_path = dataset_dir / "gps_interpolated_with_orientation.csv"

        if not all(
            [real_skyline_dir.is_dir(), dem_skyline_dir.is_dir(), gps_path.is_file()]
        ):
            raise FileNotFoundError(f"Required data not found in {dataset_dir}.")

        gps_df = pd.read_csv(gps_path)

        all_anchor_paths = []
        all_positive_paths = []

        if mode == "train":
            print("Loading and verifying all available pairs...")

        for _, row in gps_df.iterrows():
            image_number = Path(row.filename).stem
            anchor_path = real_skyline_dir / f"binary_panorama_{image_number}.png"
            positive_path = dem_skyline_dir / f"{image_number}.npy"

            if anchor_path.exists() and positive_path.exists():
                all_anchor_paths.append(anchor_path)
                all_positive_paths.append(positive_path)

        if not all_anchor_paths:
            raise ValueError(f"No valid anchor/positive pairs found in {dataset_dir}")

        train_anchors, val_anchors, train_positives, val_positives = train_test_split(
            all_anchor_paths,
            all_positive_paths,
            test_size=val_split,
            random_state=random_state,
        )

        if self.mode == "train":
            self.anchor_paths = train_anchors
            self.positive_paths = train_positives
        elif self.mode == "val":
            self.anchor_paths = val_anchors
            self.positive_paths = val_positives
        else:
            raise ValueError(
                f"Mode '{self.mode}' not recognized. Use 'train' or 'val'."
            )

        # The negative pool should always be the training anchors to avoid data leakage
        self.negative_pool = train_anchors
        self.num_samples = len(self.anchor_paths)
        print(
            f"Loaded {len(all_anchor_paths)} total pairs. Using {self.num_samples} for '{self.mode}' mode."
        )

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
            negative_path_str = random.choice(self.negative_pool)
            if negative_path_str != anchor_path:
                break
        negative_img = Image.open(negative_path_str).convert("L")

        # 2. Apply transformations
        if self.transform:
            if self.mode == "train":
                # --- Apply consistent random augmentations for the positive pair ---
                # 1. Random Horizontal Flip
                if torch.rand(1) < 0.5:
                    anchor_img = F.hflip(anchor_img)
                    positive_img = F.hflip(positive_img)

                # 2. Random Affine (rotation, translation, scale)
                affine_params = transforms.RandomAffine.get_params(
                    degrees=(-5, 5),
                    translate=(0.05, 0.05),
                    scale_ranges=(0.95, 1.05),
                    shears=None,
                    img_size=list(anchor_img.size),
                )
                anchor_img = F.affine(anchor_img, *affine_params)
                positive_img = F.affine(positive_img, *affine_params)

                # --- Apply independent augmentations to the negative image ---
                # The full transform pipeline is applied here.
                negative_tensor = self.transform(negative_img)

                # --- Manually apply the rest of the pipeline to the positive pair ---
                # This is more robust than slicing the transform list.
                resize = transforms.Resize((224, 224), antialias=True)
                to_tensor = transforms.ToTensor()
                repeater = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                normalizer = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

                # Apply to anchor
                anchor_img = resize(anchor_img)
                anchor_tensor = to_tensor(anchor_img)
                anchor_tensor = repeater(anchor_tensor)
                anchor_tensor = normalizer(anchor_tensor)

                # Apply to positive
                positive_img = resize(positive_img)
                positive_tensor = to_tensor(positive_img)
                positive_tensor = repeater(positive_tensor)
                positive_tensor = normalizer(positive_tensor)

                return anchor_tensor, positive_tensor, negative_tensor

            else:  # self.mode == 'val'
                # For validation, apply the simple, non-random transform to all
                return (
                    self.transform(anchor_img),
                    self.transform(positive_img),
                    self.transform(negative_img),
                )

        return anchor_img, positive_img, negative_img
