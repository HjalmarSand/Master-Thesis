import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

#TODO: Investigate some things, firstly, we are using cosine similarity it is probably better to use a triplet loss.  
#Secondly, we probably want to try investigating the 

IMG_DIR = "images/"
IMG_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
VALIDATION_SPLIT = 0.2


class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNetwork, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        # L2 normalize the embedding - important for cosine similarity
        embedding = F.normalize(self.projection_head(features), p=2, dim=1)
        return embedding


class TwoTowerMetricLearner(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TwoTowerMetricLearner, self).__init__()
        self.dem_tower = EmbeddingNetwork(embedding_dim)
        self.cam_tower = EmbeddingNetwork(embedding_dim)

    def forward(self, dem_input, cam_input):
        dem_embedding = self.dem_tower(dem_input)
        cam_embedding = self.cam_tower(cam_input)
        return dem_embedding, cam_embedding


def distance_to_target_similarity(dist, max_dist=1000.0):
    dist = torch.clamp(dist, 0, max_dist)
    return 1.0 - (dist / max_dist)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, max_dist=2000.0):
        super(CosineSimilarityLoss, self).__init__()
        self.max_dist = max_dist
        self.cos = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss()

    def forward(self, emb1, emb2, true_distances):
        actual_similarity = self.cos(emb1, emb2)
        target_similarity = distance_to_target_similarity(
            true_distances.to(emb1.device), self.max_dist
        )
        loss = self.mse_loss(actual_similarity, target_similarity)
        return loss


class SkylinePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_points = self._load_data_points()

    def _load_data_points(self):
        points = []
        print("Loading data points for metric learning...")
        for dataset_name in sorted(os.listdir(self.root_dir)):
            dataset_path = os.path.join(self.root_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            print(f"  - Processing {dataset_name}")
            csv_path = os.path.join(
                dataset_path, "gps_interpolated_with_orientation.csv"
            )
            dem_dir = os.path.join(dataset_path, "dem_skylines_paired")
            cam_dir = os.path.join(dataset_path, "binary_panoramas_aligned")
            if not all(os.path.exists(p) for p in [csv_path, dem_dir, cam_dir]):
                print(f"    - Skipping {dataset_name}, missing required files/dirs.")
                continue
            df = pd.read_csv(csv_path)
            dem_map = {
                re.search(r"(\d{4})", f).group(1): os.path.join(dem_dir, f)
                for f in os.listdir(dem_dir)
            }
            cam_map = {
                re.search(r"(\d{4})", f).group(1): os.path.join(cam_dir, f)
                for f in os.listdir(cam_dir)
            }
            for _, row in df.iterrows():
                img_num_str = f"{int(row['n']):04d}"
                if img_num_str in dem_map and img_num_str in cam_map:
                    points.append(
                        {
                            "dem_path": dem_map[img_num_str],
                            "cam_path": cam_map[img_num_str],
                            "pos": (row["easting"], row["northing"]),
                        }
                    )
        if not points:
            raise RuntimeError(
                "No valid data points found! Check your directory structure and file names."
            )
        print(f"Found {len(points)} valid data points.")
        return points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        anchor_point = self.data_points[idx]
        other_idx = random.randint(0, len(self.data_points) - 1)
        other_point = self.data_points[other_idx]
        anchor_dem_img = Image.open(anchor_point["dem_path"]).convert("L")
        other_cam_img = Image.open(other_point["cam_path"]).convert("L")
        if self.transform:
            anchor_dem_img = self.transform(anchor_dem_img)
            other_cam_img = self.transform(other_cam_img)
        pos_anchor = np.array(anchor_point["pos"])
        pos_other = np.array(other_point["pos"])
        true_distance = np.linalg.norm(pos_anchor - pos_other)
        return (
            anchor_dem_img,
            other_cam_img,
            torch.tensor(true_distance, dtype=torch.float32),
        )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_dataset = SkylinePairDataset(root_dir=IMG_DIR, transform=transform)

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = TwoTowerMetricLearner(embedding_dim=128).to(device)
    criterion = CosineSimilarityLoss(max_dist=5000.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for anchor_dem_masks, other_cam_masks, distances in train_pbar:
            anchor_dem_masks = anchor_dem_masks.to(device)
            other_cam_masks = other_cam_masks.to(device)
            distances = distances.to(device)
            optimizer.zero_grad()
            dem_embeddings, cam_embeddings = model(anchor_dem_masks, other_cam_masks)
            loss = criterion(dem_embeddings, cam_embeddings, distances)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * anchor_dem_masks.size(0)
            train_pbar.set_postfix({"loss": loss.item()})

        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")
            for anchor_dem_masks, other_cam_masks, distances in val_pbar:
                anchor_dem_masks = anchor_dem_masks.to(device)
                other_cam_masks = other_cam_masks.to(device)
                distances = distances.to(device)
                dem_embeddings, cam_embeddings = model(
                    anchor_dem_masks, other_cam_masks
                )
                loss = criterion(dem_embeddings, cam_embeddings, distances)
                running_val_loss += loss.item() * anchor_dem_masks.size(0)
                val_pbar.set_postfix({"loss": loss.item()})

        epoch_val_loss = running_val_loss / len(val_dataset)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_metric_model.pth")
            print(
                f"  -> New best metric model saved with validation loss: {best_val_loss:.4f}"
            )

    print("Training finished.")
