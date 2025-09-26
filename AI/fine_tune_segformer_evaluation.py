import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from fine_tune_segformer import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SkySegmentationDataset,
    parent_dataset_name,
    random_split_per_dataset,
    val_tf,  # reuse your CenterCrop(640x1280)
)
from PIL import Image
from torch.utils.data import DataLoader, Subset

import constants

# --- import your training code helpers ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINT_BASE = "nvidia/segformer-b5-finetuned-ade-640-640"
FINETUNED_DIR = (
    constants.PROJECT_ROOT / "AI/models/segformer-b5-sky-finetuned-multi-dataset"
)
DATASET_NAMES = ["dataset1", "dataset2", "dataset3", "dataset4"]
BATCH_SIZE = 4
THRESH = 0.5
EPS = 1e-9


def discover_pairs():
    pairs = []
    for name in DATASET_NAMES:
        base_image_dir = constants.IMAGES_DIR / name
        base_annotation_dir = (
            constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations" / name
        )
        if not base_image_dir.is_dir() or not base_annotation_dir.is_dir():
            continue
        for image_subfolder in base_image_dir.iterdir():
            if image_subfolder.is_dir() and image_subfolder.name.startswith(
                "annotated_panoramas_"
            ):
                suffix = image_subfolder.name.removeprefix("annotated_panoramas_")
                ann_file = base_annotation_dir / f"manual_annotations_{suffix}.json"
                if ann_file.is_file():
                    pairs.append(
                        {"image_dir": image_subfolder, "annotation_file": ann_file}
                    )
    if not pairs:
        raise RuntimeError("No valid dataset pairs found. Check folder structure.")
    return pairs


def accumulate_item(stats, key, tp, fp, fn, tn):
    stats[key]["TP"] += tp
    stats[key]["FP"] += fp
    stats[key]["FN"] += fn
    stats[key]["TN"] += tn


def finalize_metrics(counts):
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    denom_iou = TP + FP + FN
    iou = TP / (denom_iou + EPS)
    pixel_acc = (TP + TN) / (TP + FP + FN + TN + EPS)
    precision = TP / (TP + FP + EPS)
    recall = TP / (TP + FN + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    return {
        "IoU": iou,
        "Pixel Accuracy": pixel_acc,
        "Recall": recall,
        "Precision": precision,
        "F1-score": f1,
    }


def run():
    pairs = discover_pairs()

    # Build full datasets (same transforms & processor as training)
    processor_base = SegformerImageProcessor.from_pretrained(
        MODEL_CHECKPOINT_BASE, reduce_labels=False
    )
    full_val = SkySegmentationDataset(pairs, processor_base, transform=val_tf)

    # Recreate split with the same seed/ratio as during training (So we evaluate on the validation set)
    full_train = SkySegmentationDataset(
        pairs, processor_base, transform=val_tf
    )  # only to reproduce indices
    train_idx, val_idx = random_split_per_dataset(full_train, val_ratio=0.2, seed=42)
    val_subset = Subset(full_val, val_idx)

    # Precompute dataset name for each item in val_subset order
    val_names = [
        parent_dataset_name(val_subset.dataset.samples[i]["image_path"])
        for i in val_subset.indices
    ]

    def eval_with_names(model, processor):
        dl = DataLoader(val_subset, batch_size=BATCH_SIZE)
        model.eval().to(DEVICE)

        # accumulates our stats
        stats_by_ds = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        overall = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

        # cursor to map batch items -> val_names
        cursor = 0
        with torch.no_grad():
            for batch in dl:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)  # [B,H,W]
                B = labels.shape[0]

                outputs = model(pixel_values=pixel_values, labels=None)
                logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # [B,H,W]
                probs = torch.sigmoid(logits)
                preds = probs > THRESH

                valid = labels != 255
                gt = (labels == 1) & valid
                pd = preds & valid

                # split per item
                for b in range(B):
                    name = val_names[cursor]
                    cursor += 1
                    TP = (pd[b] & gt[b]).sum().item()
                    FP = (pd[b] & ~gt[b]).sum().item()
                    FN = ((~pd[b]) & gt[b]).sum().item()
                    TN = ((~pd[b]) & (~gt[b]) & valid[b]).sum().item()

                    accumulate_item(stats_by_ds, name, TP, FP, FN, TN)
                    accumulate_item({"overall": overall}, "overall", TP, FP, FN, TN)

        # finalize per dataset + overall
        metrics_by_ds = {k: finalize_metrics(v) for k, v in stats_by_ds.items()}
        metrics_overall = finalize_metrics(overall)
        return metrics_by_ds, metrics_overall

    # Using off-the-shelf segformerb5
    base_processor = SegformerImageProcessor.from_pretrained(
        MODEL_CHECKPOINT_BASE, reduce_labels=False
    )
    base_model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT_BASE,
        num_labels=1,
        id2label={0: "sky"},
        label2id={"sky": 0},
        ignore_mismatched_sizes=True,
    )

    # Load our fine-tuned segformer-b5
    ft_processor = SegformerImageProcessor.from_pretrained(
        FINETUNED_DIR, reduce_labels=False
    )
    ft_model = SegformerForSemanticSegmentation.from_pretrained(
        FINETUNED_DIR,
        num_labels=1,
        id2label={0: "sky"},
        label2id={"sky": 0},
        ignore_mismatched_sizes=True,
    )

    base_by_ds, base_all = eval_with_names(base_model, base_processor)
    ft_by_ds, ft_all = eval_with_names(ft_model, ft_processor)

    # Normalize dataset order (only dataset1..dataset3 for the table)
    wanted = ["dataset1", "dataset2", "dataset3"]
    rows = []

    def fmt(x):
        return f"{100 * x:.2f}"  # as percentage with 2 decimals

    for d in wanted:
        if (d not in base_by_ds) and (d not in ft_by_ds):
            continue
        base_m = base_by_ds.get(d, None)
        ft_m = ft_by_ds.get(d, None)
        rows.append(
            {
                "Dataset": d.title().replace("Dataset", "Dataset "),
                "Model": "Segformer-b5",
                "IoU": fmt(base_m["IoU"]) if base_m else "",
                "Pixel Accuracy": fmt(base_m["Pixel Accuracy"]) if base_m else "",
                "Recall": fmt(base_m["Recall"]) if base_m else "",
                "Precision": fmt(base_m["Precision"]) if base_m else "",
                "F1-score": fmt(base_m["F1-score"]) if base_m else "",
            }
        )
        rows.append(
            {
                "Dataset": d.title().replace("Dataset", "Dataset "),
                "Model": "Segformer-b5f",
                "IoU": fmt(ft_m["IoU"]) if ft_m else "",
                "Pixel Accuracy": fmt(ft_m["Pixel Accuracy"]) if ft_m else "",
                "Recall": fmt(ft_m["Recall"]) if ft_m else "",
                "Precision": fmt(ft_m["Precision"]) if ft_m else "",
                "F1-score": fmt(ft_m["F1-score"]) if ft_m else "",
            }
        )

    # Averages (overall micro)
    rows.append(
        {
            "Dataset": "\\textbf{Average}",
            "Model": "Segformer-b5",
            "IoU": fmt(base_all["IoU"]),
            "Pixel Accuracy": fmt(base_all["Pixel Accuracy"]),
            "Recall": fmt(base_all["Recall"]),
            "Precision": fmt(base_all["Precision"]),
            "F1-score": fmt(base_all["F1-score"]),
        }
    )
    rows.append(
        {
            "Dataset": "\\textbf{Average}",
            "Model": "Segformer-b5f",
            "IoU": fmt(ft_all["IoU"]),
            "Pixel Accuracy": fmt(ft_all["Pixel Accuracy"]),
            "Recall": fmt(ft_all["Recall"]),
            "Precision": fmt(ft_all["Precision"]),
            "F1-score": fmt(ft_all["F1-score"]),
        }
    )

    # Write CSV for auditing
    out_csv = constants.PROJECT_ROOT / "AI" / "metrics_segformer_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "Model",
                "IoU",
                "Pixel Accuracy",
                "Recall",
                "Precision",
                "F1-score",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print LaTeX rows
    def latex_row(dataset, model, m):
        return f"\\multirow{{2}}{{*}}{{{dataset}}} \n  & {model} & {m['IoU']} & {m['Pixel Accuracy']} & {m['Recall']} & {m['Precision']} & {m['F1-score']} \\\\"

    print("\n==== LaTeX table body ====\n")
    # Per dataset blocks
    for d in ["Dataset 1", "Dataset 2", "Dataset 3"]:
        base_m = next(
            (r for r in rows if r["Dataset"] == d and r["Model"] == "Segformer-b5"),
            None,
        )
        ft_m = next(
            (r for r in rows if r["Dataset"] == d and r["Model"] == "Segformer-b5f"),
            None,
        )
        if base_m or ft_m:
            if base_m:
                print(latex_row(d, "Segformer-b5", base_m))
            if ft_m:
                print(
                    f"  & Segformer-b5f & {ft_m['IoU']} & {ft_m['Pixel Accuracy']} & {ft_m['Recall']} & {ft_m['Precision']} & {ft_m['F1-score']} \\\\"
                )
            print("\\midrule")

    # Average block
    base_avg = next(
        r
        for r in rows
        if r["Dataset"] == "\\textbf{Average}" and r["Model"] == "Segformer-b5"
    )
    ft_avg = next(
        r
        for r in rows
        if r["Dataset"] == "\\textbf{Average}" and r["Model"] == "Segformer-b5f"
    )
    print("\\multirow{2}{*}{\\textbf{Average}}")
    print(
        f"  & Segformer-b5 & {base_avg['IoU']} & {base_avg['Pixel Accuracy']} & {base_avg['Recall']} & {base_avg['Precision']} & {base_avg['F1-score']} \\\\"
    )
    print(
        f"  & Segformer-b5f & {ft_avg['IoU']} & {ft_avg['Pixel Accuracy']} & {ft_avg['Recall']} & {ft_avg['Precision']} & {ft_avg['F1-score']} \\\\"
    )
    print("\n==========================\n")
    print(f"CSV written to: {out_csv}")


if __name__ == "__main__":
    run()
