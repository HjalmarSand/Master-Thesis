import os
import random
import sys
from collections import Counter, defaultdict

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to Python's system path
sys.path.append(parent_dir)

import constants

train_tf = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        # A.SmallestMaxSize(max_size=640),  # height ~ 640, width ~ 3413 for 5120×960
        A.RandomCrop(height=640, width=1280),  # or 1536 if VRAM allows
        A.OneOf([A.GaussianBlur(p=0.5), A.MotionBlur(p=0.5)], p=0.2),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.3),
    ]
)

val_tf = A.Compose(
    [
        # A.SmallestMaxSize(max_size=640),
        A.CenterCrop(height=640, width=1280),  # match train crop width
    ]
)


# Phase 1: The Custom PyTorch Dataset
class SkySegmentationDataset(Dataset):
    """
    PyTorch Dataset for binary sky segmentation from multiple COCO datasets.
    - Scans multiple dataset directories.
    - For each dataset, it loads images and parses its COCO annotations.
    - Creates a master list of all samples.
    - Uses a processor to prepare data for the SegFormer model.
    """

    def __init__(self, dataset_configs, processor, transform=None):
        self.processor = processor
        self.samples = []
        self.transform = transform

        for config in tqdm(dataset_configs, desc="Processing datasets"):
            image_dir = config["image_dir"]
            annotation_file = config["annotation_file"]

            coco = COCO(annotation_file)
            image_ids = coco.getImgIds()

            sky_ids = coco.getCatIds(catNms=["sky"])
            if not sky_ids:
                raise ValueError(f"'sky' category not found in {annotation_file}")
            sky_id = sky_ids[0]

            # Create a sample for each image in the current dataset
            for image_id in image_ids:
                image_info = coco.loadImgs(image_id)[0]
                image_path = os.path.join(image_dir, image_info["file_name"])

                # Store all necessary info for this one sample
                self.samples.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "image_info": image_info,
                        "coco_instance": coco,
                        "sky_id": sky_id,
                        "group": os.path.basename(image_dir),
                        "dataset": os.path.basename(
                            os.path.dirname(image_dir)
                        ),  # e.g. dataset1
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Get image info and load the image
        sample = self.samples[idx]
        image_id = sample["image_id"]
        image_info = sample["image_info"]
        image_path = sample["image_path"]
        coco = sample["coco_instance"]
        # Convert to numpy array for albumentations
        image = np.array(Image.open(image_path).convert("RGB"))

        # 2. Create the binary mask
        mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=[sample["sky_id"]])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            mask[coco.annToMask(ann) == 1] = 1

        # 3. Apply augmentations IF a transform is provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        inputs = self.processor(images=image, return_tensors="pt", do_resize=False)
        pixel_values = inputs["pixel_values"].squeeze()
        labels = torch.from_numpy(mask).long()  # {0,1}
        return {"pixel_values": pixel_values, "labels": labels}


def random_split_per_dataset(ds, val_ratio=0.2, seed=42, min_train=1, min_val=1):
    """
    For each dataset name in ds.samples, do an independent random 80/20 split,
    then concatenate all per-dataset splits to form global train/val indices.
    """
    rng = random.Random(seed)

    # bucket indices by dataset (ignores 'group'/annotator)
    by_dataset = defaultdict(list)
    for idx, s in enumerate(ds.samples):
        by_dataset[s["dataset"]].append(idx)

    train_idx, val_idx = [], []
    for dname, idxs in by_dataset.items():
        idxs = idxs[:]  # copy
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(min_val, int(round(n * val_ratio)))
        n_val = (
            min(n_val, n - min_train) if n > min_train else 0
        )  # keep at least min_train
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def parent_dataset_name(
    p,
):  # assumes .../images/<DATASET_NAME>/annotated_panoramas_*/<files>
    parts = str(p).split(os.sep)
    # find "images" then take next part as dataset name
    if "images" in parts:
        i = parts.index("images")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def split_summary(subset, name):
    names = [
        parent_dataset_name(subset.dataset.samples[i]["image_path"])
        for i in subset.indices
    ]
    c = Counter(names)
    print(f"{name} size: {len(subset)} | per-dataset: {dict(c)}")


def estimate_pos_weight(subset, sample_k=200):
    import random

    import torch

    idxs = list(range(len(subset)))
    random.shuffle(idxs)
    idxs = idxs[: min(sample_k, len(idxs))]
    pos = neg = 0
    for i in idxs:
        y = subset[i]["labels"]  # [H,W] with {0,1,255}
        valid = y != 255
        pos += (y[valid] == 1).sum().item()
        neg += (y[valid] == 0).sum().item()
    if pos == 0:  # avoid div by zero; fall back to neutral weight
        return torch.tensor([1.0])
    return torch.tensor([neg / pos])


# Phase 2: Model Preparation

# A. Define Model and Label Configuration
model_checkpoint = "nvidia/segformer-b5-finetuned-ade-640-640"  # TODO: Maybe we want to finetune a smaller model to not have it overfit in its training as esaily with a smal dataset. I think it is learning to memorize specifix pixel patterns that do not generalize well.
num_labels = 1
id2label = {0: "sky"}
label2id = {"sky": 0}

processor = SegformerImageProcessor.from_pretrained(
    model_checkpoint, reduce_labels=False
)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Freeze backbone; train full decode head (unchanged from your intent)
for _, p in model.named_parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if name.startswith("decode_head."):
        p.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
percentage_trainable = (trainable_params / total_params) * 100

print("\nParameter Count Verification")
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters:   {trainable_params:,}")
print(f"Percentage trainable:   {percentage_trainable:.2f}%")

# 1. Main Configuration
# Define the datasets to include in the training
print("\nDynamically Discovering Paired Datasets")
# Define the main dataset groups you want to search within
DATASET_NAMES = ["dataset1", "dataset2", "dataset3", "dataset4"]

# This list will hold the final configurations for the Dataset class
dataset_configs = []

# Loop through each main dataset group (e.g., "dataset1")
for name in DATASET_NAMES:
    # Path to the directory containing the image subfolders (e.g., .../images/dataset1/)
    base_image_dir = constants.IMAGES_DIR / name

    # Path to the directory containing the annotation files (e.g., .../annotations/dataset1/)
    base_annotation_dir = (
        constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations" / name
    )

    # Robustness Check: Ensure the base directories exist
    if not base_image_dir.is_dir():
        print(
            f"Warning: Base image directory for '{name}' not found, skipping: {base_image_dir}"
        )
        continue
    if not base_annotation_dir.is_dir():
        print(
            f"Warning: Base annotation directory for '{name}' not found, skipping: {base_annotation_dir}"
        )
        continue

    # Discover Image Subfolders and Find Their Paired Annotation File
    print(f"Searching for paired image/annotation sources in group '{name}'...")

    # Iterate over every item in the base image directory
    for image_subfolder in base_image_dir.iterdir():
        # We only care about directories whose names start with the correct prefix
        if image_subfolder.is_dir() and image_subfolder.name.startswith(
            "annotated_panoramas_"
        ):
            # This is the key logic change
            # 1. Derive the unique suffix from the image folder's name.
            #    Example: "annotated_panoramas_dataset1_arvid" -> "dataset1_arvid"
            suffix = image_subfolder.name.removeprefix("annotated_panoramas_")

            # 2. Construct the full path for the expected matching annotation file.
            #    Example: .../annotations/dataset1/manual_annotations_dataset1_arvid.json
            expected_annotation_file = (
                base_annotation_dir / f"manual_annotations_{suffix}.json"
            )

            # 3. Final Robustness Check: Does this specific annotation file actually exist?
            if expected_annotation_file.is_file():
                # If we found the image folder AND its matching annotation file, it's a valid pair.
                print("  [+] Found valid pair:")
                print(f"      - Image Dir:    {image_subfolder.name}")
                print(f"      - Annotation:   {expected_annotation_file.name}")

                dataset_configs.append(
                    {
                        "image_dir": image_subfolder,
                        "annotation_file": expected_annotation_file,
                    }
                )
            else:
                # If the image folder exists but its annotation file is missing, we print a warning and skip it.
                print(
                    f"  [!] Found image folder '{image_subfolder.name}', but its matching annotation file was NOT found at: {expected_annotation_file}"
                )

print(
    f"\nDiscovery Complete: Found {len(dataset_configs)} total valid data pairs to train on."
)

# Add a final critical check in case no data was found at all
if not dataset_configs:
    print(
        "\nCRITICAL ERROR: No valid datasets could be loaded. Please check your folder structure and file names."
    )
    print("Expected structure example:")
    print("  - .../images/dataset1/annotated_panoramas_dataset1_arvid/")
    print("  - .../annotations/dataset1/manual_annotations_dataset1_arvid.json")
    import sys

    sys.exit(1)

OUTPUT_DIR = (
    constants.PROJECT_ROOT / "AI/models/segformer-b5-sky-finetuned-multi-dataset"
)
SKY_CATEGORY_ID = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

# 2. Create Datasets and Dataloaders
full_train = SkySegmentationDataset(dataset_configs, processor, transform=train_tf)
full_val = SkySegmentationDataset(dataset_configs, processor, transform=val_tf)

train_indices, val_indices = random_split_per_dataset(
    full_train, val_ratio=0.2, seed=42
)

train_subset = Subset(full_train, train_indices)
val_subset = Subset(full_val, val_indices)

split_summary(train_subset, "TRAIN")
split_summary(val_subset, "VAL")

print("unique samples:", len(full_train))
print("overlap:", len(set(train_subset.indices) & set(val_subset.indices)))
print("cover:", len(set(train_subset.indices) | set(val_subset.indices)))


pos_weight = estimate_pos_weight(train_subset).to(DEVICE)
print(f"pos_weight for BCE: {pos_weight.item():.3f}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE)

b = next(iter(val_dataloader))
print(
    "uniques in val labels:", sorted([int(v) for v in torch.unique(b["labels"])])
)  # expect [0,1] (no 255)
print("sky pixel ratio:", ((b["labels"] == 1).float().mean().item()))
print("val batch label shape:", b["labels"].shape)

# 3. Setup Optimizer and Metric
model.to(DEVICE)
# Passing only the parameters that are trainable to the optimizer
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",  # because we want to maximize mIoU
    factor=0.5,  # reduce LR by half
    patience=2,  # wait 2 epochs with no improvement
)

# 4. The Training Loop
best_score = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # TRAIN
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)  # {0,1,255}

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=None)  # we'll compute loss
        # logits: [B, 1, h, w]
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)  # -> [B, H, W]

        valid = labels != 255
        y = (labels == 1).float()
        if valid.any().item():
            loss = criterion(logits[valid], y[valid])
            loss.backward()
            optimizer.step()

    # VALIDATE
    model.eval()
    tot_intersection = 0
    tot_union = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=None)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B,H,W]

            probs = torch.sigmoid(logits)
            preds = probs > 0.5  # sky prediction
            gt = labels == 1
            valid = labels != 255

            # mask invalid pixels
            preds = preds & valid
            gt = gt & valid

            intersection = (preds & gt).sum().item()
            union = (preds | gt).sum().item()
            tot_intersection += intersection
            tot_union += union

    # sky IoU (handle empty-union case)
    sky_iou = (tot_intersection / tot_union) if tot_union > 0 else 0.0

    print(f"  Validation Sky IoU: {sky_iou:.4f}")

    scheduler.step(sky_iou)

    if sky_iou > best_score:
        best_score = sky_iou
        print(f"  New best Sky IoU! Saving model to {OUTPUT_DIR}")
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

print(
    f"\nTraining finished! Best Sky IoU: {best_score:.4f}. Model saved to {OUTPUT_DIR}"
)
