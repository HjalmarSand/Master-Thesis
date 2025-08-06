import os
import sys

import torch
import torchmetrics
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to Python's system path
sys.path.append(parent_dir)

import constants


# Phase 1: The Custom PyTorch Dataset
class SkySegmentationDataset(Dataset):
    """
    PyTorch Dataset for binary sky segmentation.
    - Loads images from a directory.
    - Parses COCO annotations to create binary masks (0=background, 1=sky).
    - Uses a processor to prepare data for the SegFormer model.
    """

    def __init__(self, root_dir, annotation_file, processor, sky_category_id):
        self.root_dir = root_dir
        self.processor = processor
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.sky_category_id = sky_category_id

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Get image info and load the image
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # 2. Create a binary segmentation mask (0s and 1s)
        mask = torch.zeros(
            (image_info["height"], image_info["width"]), dtype=torch.uint8
        )
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[self.sky_category_id])
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            # Draw the polygon for the "sky" annotation onto the mask with a value of 1
            mask[self.coco.annToMask(ann) == 1] = 1

        # 3. Use the processor to prepare the image and mask
        inputs = self.processor(
            images=image, segmentation_maps=mask, return_tensors="pt"
        )

        # The processor nests outputs, so we squeeze to remove the batch dimension
        pixel_values = inputs["pixel_values"].squeeze()
        labels = inputs["labels"].squeeze()

        return {"pixel_values": pixel_values, "labels": labels}


# Phase 2: Model Preparation

# A. Define Model and Label Configuration
model_checkpoint = "nvidia/segformer-b5-finetuned-ade-640-640"  # TODO: Maybe we want to finetune a smaller model to not have it overfit in its training as esaily with a smal dataset. I think it is learning to memorize specifix pixel patterns that do not generalize well.
num_labels = 2
id2label = {0: "sky", 1: "background"}  # sky is class 0, background is class 1
label2id = {v: k for k, v in id2label.items()}

# B. Load Processor and Model
processor = SegformerImageProcessor.from_pretrained(
    model_checkpoint, reduce_labels=True
)
model = SegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # Allows loading the backbone with a new head
)

# C. Freeze the Backbone Encoder
# for name, param in model.named_parameters():
#    if name.startswith("segformer."):
#        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("decode_head."):
        print(name)

for name, param in model.named_parameters():
    # If the parameter's name is NOT in the classifier, freeze it
    if "decode_head.classifier" not in name:
        param.requires_grad = False
    # The only thing left as trainable is decode_head.classifier

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
percentage_trainable = (trainable_params / total_params) * 100

print("\nParameter Count Verification")
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters:   {trainable_params:,}")
print(f"Percentage trainable:   {percentage_trainable:.2f}%")

# Phase 3: The Training Script

# 1. Main Configuration
# NOTE: Update these paths depending on the dataset!
IMAGE_DIR = constants.IMAGES_DIR / "dataset2/panoramas"
ANNOTATION_FILE = (
    constants.PROJECT_ROOT
    / "AI/segformer_segmentation_annotations/dataset2/manual_annotations.json"
)
OUTPUT_DIR = constants.PROJECT_ROOT / "AI/models/segformer-b5-sky-finetuned"
SKY_CATEGORY_ID = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

# 2. Create Datasets and Dataloaders
full_dataset = SkySegmentationDataset(
    root_dir=IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    processor=processor,
    sky_category_id=SKY_CATEGORY_ID,
)

# Simple 80/20 split for training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 3. Setup Optimizer and Metric
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
metric = torchmetrics.JaccardIndex(
    task="multiclass", num_classes=2, ignore_index=255
).to(DEVICE)

# 4. The Training Loop
best_miou = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training phase
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)

            # Get predictions (logits) and upscale them to match label size
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            # Get the predicted class (0 or 1)
            predicted = upsampled_logits.argmax(dim=1)

            # Update the metric
            metric.update(predicted, labels)

    # Calculate and print metrics for the epoch
    miou = metric.compute()
    metric.reset()  # Reset metric for the next epoch

    print(f"  Validation mIoU: {miou:.4f}")

    # 5. Save the best model based on validation mIoU
    if miou > best_miou:
        best_miou = miou
        print(f"  New best mIoU! Saving model to {OUTPUT_DIR}")
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

print(
    f"\nTraining finished! Best model saved to {OUTPUT_DIR} with mIoU: {best_miou:.4f}"
)
