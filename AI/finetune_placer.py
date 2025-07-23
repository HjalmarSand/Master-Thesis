import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

import constants
from AI.triplet_dataset import SkylineTripletDataset


class EmbeddingNet(nn.Module):
    """
    A network to generate embeddings from skyline images.
    It uses a pre-trained MobileNetV2, modified only at the final layer for embedding output.
    """

    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Load the default pre-trained MobileNetV2
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        num_features = self.backbone.last_channel
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.backbone(x)


def train_placer_model(
    dataset_dir: Path,
    model_save_path: Path,
    num_epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    """
    Main function to run the training process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define the transformations
    image_transform = transforms.Compose(
        [
            # This transform randomly crops the image while varying the aspect ratio
            # before resizing, making the model more robust to distortion.
            transforms.RandomResizedCrop(
                size=(224, 224), scale=(0.8, 1.0), ratio=(1.3, 5.5), antialias=True
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2. Initialize the Dataset and DataLoader
    # Note: For training, we use the full panorama. The learned features will then be
    # extracted from windows during inference
    triplet_dataset = SkylineTripletDataset(
        dataset_dir=dataset_dir, transform=image_transform
    )
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize the Model, Loss, and Optimizer
    model = EmbeddingNet(embedding_dim=128).to(device)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Starting model fine-tuning...")
    # 4. The Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Summary - Average Loss: {epoch_loss:.4f}")

    # 5. Save the fine-tuned model weights
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    # We save the entire model's state dict, which includes the backbone and new classifier
    torch.save(model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved to: {model_save_path}")


def run_finetuning_stage():
    """
    Wrapper function to be called as a stage from main.py.
    """
    try:
        # Find available datasets that have the required data
        datasets_root_dir = Path(constants.IMAGES_DIR)
        available_datasets = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "dem_skylines_paired").is_dir()
            ]
        )

        if not available_datasets:
            print("No datasets found with paired DEM skylines needed for training.")
            return False

        # Interactive selection of the dataset
        selected_dir = None
        while True:
            print("\nPlease select a dataset to use for fine-tuning:")
            for i, dir_path in enumerate(available_datasets):
                print(f"  [{i}]: {dir_path.name}")
            choice_str = input("Enter the number of the dataset: ")
            try:
                choice_idx = int(choice_str)
                if 0 <= choice_idx < len(available_datasets):
                    selected_dir = available_datasets[choice_idx]
                    break
                else:
                    print("Invalid input.")
            except ValueError:
                print("Invalid input.")

        print(f"\nTraining on dataset: {selected_dir.name}")
        project_root = Path(__file__).resolve().parent.parent
        model_save_path = (
            project_root / "AI" / "models" / f"finetuned_placer_{selected_dir.name}.pth"
        )

        train_placer_model(
            dataset_dir=selected_dir,
            model_save_path=model_save_path,
            num_epochs=25,
            lr=1e-4,
            batch_size=16,
        )
        return True
    except Exception as e:
        print(f"An error occurred during model fine-tuning: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_finetuning_stage()
