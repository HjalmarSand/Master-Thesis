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
    def __init__(self, embedding_dim=128, freeze_layers=19):
        super(EmbeddingNet, self).__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        for i, child in enumerate(self.backbone.features.children()):
            if i < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False

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
    num_epochs: int,
    lr: float,
    batch_size: int,
    patience: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SkylineTripletDataset(
        dataset_dir=dataset_dir, transform=train_transform, mode="train"
    )
    val_dataset = SkylineTripletDataset(
        dataset_dir=dataset_dir, transform=val_transform, mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = EmbeddingNet(embedding_dim=128).to(device)
    loss_fn = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = model_save_path.with_name(f"{model_save_path.stem}_best.pth")

    if best_model_path.exists():
        print(f"Found existing best model at {best_model_path}. Resuming training.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No existing best model found. Starting training from scratch.")

    print("Starting model fine-tuning with validation...")
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
        )
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            anchor_embedding, positive_embedding, negative_embedding = (
                model(anchor),
                model(positive),
                model(negative),
            )
            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
            ):
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )
                anchor_embedding, positive_embedding, negative_embedding = (
                    model(anchor),
                    model(positive),
                    model(negative),
                )
                loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Validation loss improved. Saved best model to {best_model_path.name}"
            )
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(
        f"\nTraining complete. Best model saved to {best_model_path.name} with validation loss: {best_val_loss:.4f}"
    )


def run_finetuning_stage():
    try:
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
            num_epochs=50,
            lr=1e-5,  # Use a low learning rate for continued refinement
            batch_size=32,
            patience=10,
        )
        return True
    except Exception as e:
        print(f"An error occurred during model fine-tuning: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_finetuning_stage()
