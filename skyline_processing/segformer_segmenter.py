import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from PIL import Image
from scipy import ndimage as ndi

import constants


def enforce_top_sky_bottom_ground(
    mask_u8: np.ndarray, connectivity: int = 4
) -> np.ndarray:
    """
    Enforce:
      - Any white component NOT touching the TOP row becomes black.
      - Any black component NOT touching the BOTTOM row becomes white.
    Input:  HxW uint8 mask (255=sky, 0=ground)
    Output: HxW uint8 mask (255=sky, 0=ground)
    """
    assert mask_u8.dtype == np.uint8
    sky0 = mask_u8 >= 128  # True = white (sky), False = black (ground)
    ground0 = ~sky0
    H, W = sky0.shape

    # 4- or 8-connectivity struct (1=4-connected, 2=8-connected in SciPy)
    conn_rank = 1 if connectivity == 4 else 2
    struct = ndi.generate_binary_structure(2, conn_rank)

    # sky: keep only components that touch the TOP
    lab_s, n_s = ndi.label(sky0, structure=struct)
    if n_s > 0:
        top_labels = np.unique(lab_s[0, :])
        top_labels = top_labels[top_labels != 0]  # drop background label 0
        sky_keep = np.isin(lab_s, top_labels)
    else:
        sky_keep = np.zeros_like(sky0, dtype=bool)

    # ground: keep only components that touch the BOTTOM
    lab_g, n_g = ndi.label(ground0, structure=struct)
    if n_g > 0:
        bot_labels = np.unique(lab_g[-1, :])
        bot_labels = bot_labels[bot_labels != 0]  # drop background label 0
        ground_keep = np.isin(lab_g, bot_labels)
    else:
        ground_keep = np.zeros_like(ground0, dtype=bool)

    # Apply the rules directly on the ORIGINAL mask:
    # - sky not top-connected -> black
    # - ground not bottom-connected -> white
    sky_final = sky_keep | (ground0 & (~ground_keep))

    return np.where(sky_final, 255, 0).astype(np.uint8)


class SemanticSegmenter:
    """
    Handles semantic segmentation of images using a tiling approach.
    Can process a directory of images or launch an interactive viewer for a single image.
    """

    def __init__(self, model_name_or_path="nvidia/segformer-b5-finetuned-ade-640-640"):
        print("Initializing Semantic Segmenter...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None
        self.id2label = {}
        self.num_classes = 0
        self.sky_id: Optional[int] = None
        self.is_binary = False
        self.default_tile_h, self.default_tile_w = 640, 1280

        try:
            # Local folder or Hub id work the same:
            self.processor = SegformerImageProcessor.from_pretrained(
                self.model_name_or_path
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name_or_path
            ).to(self.device)
            self.model.eval()

            self.id2label = self.model.config.id2label
            self.num_classes = self.model.config.num_labels
            self.is_binary = self.num_classes == 1
            self.default_tile_h, self.default_tile_w = 640, 1280

            if not self.is_binary:
                self._find_sky_id()

            print(f"Default tile (h,w): {self.default_tile_h}x{self.default_tile_w}")
            print(f"Using device: {self.device}")
            print("Model and processor loaded successfully.")
        except Exception as e:
            print(
                f"CRITICAL ERROR: Failed to load model '{self.model_name_or_path}'. Exception: {e}"
            )

    def _infer_processor_size(self) -> Tuple[int, int]:
        """
        Infer (height, width) from the processor's config.
        Falls back to (640, 640) if unspecified.
        """
        size = getattr(self.processor, "size", {}) or {}
        # HF SegFormer preprocessors commonly store {"height": H, "width": W}
        h = size.get("height", size.get("shortest_edge", 640))
        w = size.get("width", 640)
        return int(h), int(w)

    def _find_sky_id(self):
        """Helper method to find the integer ID for the 'sky' label."""
        # id2label can have int or str keys
        for an_id, label in self.id2label.items():
            try:
                _id = int(an_id)
            except Exception:
                _id = an_id
            if isinstance(label, str) and "sky" in label.lower():
                self.sky_id = int(_id)
                print(f"Found sky label '{label}' with ID: {self.sky_id}")
                return

        # Fallback: if 2-class model and label names are generic, assume class 1 is sky
        if self.model and self.model.config.num_labels == 2:
            print(
                "Warning: No 'sky' label string found; assuming ID=1 for sky (binary model)."
            )
            self.sky_id = 1
        else:
            print("Warning: Could not find a 'sky' label in the model's configuration.")

    def _segment_single_image(self, image_path, tile_h=None, tile_w=None, overlap=64):
        if not self.model or not os.path.exists(image_path):
            print(f"Error: Model not loaded or image not found at {image_path}")
            return None

        tile_h = tile_h or self.default_tile_h
        tile_w = tile_w or self.default_tile_w

        original_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(original_pil)
        img_h, img_w, _ = image_np.shape

        if self.is_binary:
            full_logit_map = np.zeros(
                (img_h, img_w), dtype=np.float32
            )  # <-- logits only
            stitch_counts = np.zeros((img_h, img_w), dtype=np.float32)
        else:
            full_logits_map = np.zeros(
                (img_h, img_w, self.num_classes), dtype=np.float32
            )
            stitch_counts = np.zeros((img_h, img_w, 1), dtype=np.float32)

        step_h = max(1, tile_h - overlap)
        step_w = max(1, tile_w - overlap)

        total_tiles_h = ((img_h - 1) // step_h) + 1
        total_tiles_w = ((img_w - 1) // step_w) + 1

        pbar = tqdm(
            total=total_tiles_h * total_tiles_w,
            desc=f"Segmenting {os.path.basename(image_path)}",
            leave=False,
        )

        for y in range(0, img_h, step_h):
            for x in range(0, img_w, step_w):
                y_end = min(y + tile_h, img_h)
                x_end = min(x + tile_w, img_w)
                y_start = max(0, y_end - tile_h)
                x_start = max(0, x_end - tile_w)
                tile = image_np[y_start:y_end, x_start:x_end, :]

                inputs = self.processor(
                    images=tile, return_tensors="pt", do_resize=False
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits
                # make sure shape is [B,C,H,W]; HF can return [B,H,W] for C==1
                if logits.ndim == 3:  # [B,H,W]
                    logits = logits.unsqueeze(1)  # -> [B,1,H,W]
                elif logits.ndim == 2:  # [H,W]
                    logits = logits.unsqueeze(0).unsqueeze(0)

                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=tile.shape[:2], mode="bilinear", align_corners=False
                )  # [B,C,h,w]

                if self.is_binary:
                    # Take batch 0, channel 0 → [h,w] logits
                    logit_tile = upsampled_logits[0, 0].cpu().numpy()
                    full_logit_map[y_start:y_end, x_start:x_end] += logit_tile
                    stitch_counts[y_start:y_end, x_start:x_end] += 1.0
                else:
                    logits_to_add = (
                        upsampled_logits.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    )
                    full_logits_map[y_start:y_end, x_start:x_end, :] += logits_to_add
                    stitch_counts[y_start:y_end, x_start:x_end, :] += 1.0

                pbar.update(1)

        pbar.close()

        if self.is_binary:
            stitch_counts[stitch_counts == 0] = 1.0
            avg_logits = full_logit_map / stitch_counts  # average logits
            prob_map = 1.0 / (1.0 + np.exp(-avg_logits))  # sigmoid
            sky_mask = prob_map > 0.5  # threshold
            final_class_id_map = sky_mask.astype(np.uint8)  # {0,1}
        else:
            stitch_counts[stitch_counts == 0] = 1.0
            averaged_logits = full_logits_map / stitch_counts
            final_class_id_map = np.argmax(averaged_logits, axis=-1)  # multiclass

        return {
            "original_image_pil": original_pil,
            "class_id_map": final_class_id_map.astype(np.uint16),
        }

    def _colorize_mask(self, class_id_map):
        """
        Creates a colorized PIL image from a class ID map for visualization.
        """
        np.random.seed(42)
        max_id = int(np.max(class_id_map))
        cmap_size = max(max_id + 1, len(self.id2label))
        cmap_rgb = np.zeros((cmap_size, 3), dtype=np.uint8)
        for i in range(cmap_size):
            cmap_rgb[i] = np.random.randint(0, 255, size=3)

        color_mask = cmap_rgb[class_id_map]
        return Image.fromarray(color_mask)

    def process_directory(
        self,
        input_dir,
        output_dir_black_white,
        output_dir_overlay,
    ):
        """
        Segment all images in input_dir, save cleaned binary masks and optional overlays.
        Cleaning uses a strict connectivity rule:
        - only sky touching TOP remains white
        - only ground touching BOTTOM remains black
        """
        if not self.model:
            print("Model not loaded, cannot process directory.")
            return 1

        if output_dir_black_white:
            output_dir_black_white.mkdir(parents=True, exist_ok=True)
        if output_dir_overlay:
            output_dir_overlay.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        )
        print(f"Found {len(image_files)} images to process in {input_dir}.")

        for image_file in tqdm(image_files, desc="Overall Progress"):
            base_name = image_file.stem

            do_black_white = (
                output_dir_black_white is not None
                and not (output_dir_black_white / f"binary_{base_name}.png").exists()
            )
            do_overlay = (
                output_dir_overlay is not None
                and not (
                    output_dir_overlay / f"binary_overlay_{base_name}.png"
                ).exists()
            )

            if not do_black_white and not do_overlay:
                continue

            seg = self._segment_single_image(str(image_file))
            if not seg:
                continue

            class_map = seg["class_id_map"]
            original_pil = seg["original_image_pil"]

            # Bool sky from logits/classes
            if self.is_binary:
                sky_mask_bool = class_map.astype(bool)
            else:
                sky_mask_bool = class_map == self.sky_id

            # Apply connectivity rule
            raw_mask_u8 = sky_mask_bool.astype(np.uint8) * 255
            cleaned_mask_u8 = enforce_top_sky_bottom_ground(raw_mask_u8, connectivity=8)

            if do_black_white:
                out_bw = output_dir_black_white / f"binary_{base_name}.png"
                Image.fromarray(cleaned_mask_u8, "L").save(out_bw)

            if do_overlay:
                out_overlay = output_dir_overlay / f"binary_overlay_{base_name}.png"
                # Build RGBA overlay directly from numpy (no putalpha with arrays)
                H, W = cleaned_mask_u8.shape
                ov = np.zeros((H, W, 4), dtype=np.uint8)
                ov[cleaned_mask_u8 > 0] = [0, 0, 255, 128]  # blue, semi-transparent
                overlay_img = Image.fromarray(ov, "RGBA")
                base_rgba = original_pil.convert("RGBA")
                Image.alpha_composite(base_rgba, overlay_img).save(out_overlay)

        print("\nBatch processing complete.")
        return 0

    def view_single_image(self, image_path):
        """
        Runs segmentation on a single image and launches the interactive viewer.
        """
        if not self.model:
            print("Model not loaded, cannot view image.")
            return

        print("Running in interactive mode...")
        segmentation_data = self._segment_single_image(image_path)

        if segmentation_data:
            print("Launching interactive viewer...")
            viewer = InteractiveSegmentationViewer(
                segmentation_data["original_image_pil"],
                segmentation_data["class_id_map"],
                self.id2label,
                self.sky_id,
            )
        else:
            print("Failed to get segmentation data.")

    def package_for_cvat_from_masks(self, image_dir, mask_dir, output_cvat_zip_path):
        """
        Packages existing images and their corresponding pre-computed binary masks
        into a single ZIP file compatible with CVAT for annotation correction.
        """

        print("Searching for images and pre-computed masks...")
        image_files = sorted(
            list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        )
        mask_files = sorted(list(mask_dir.glob("*.png")))

        if not image_files:
            print(f"Error: No images found in {image_dir}.")
            return 1
        if not mask_files:
            print(f"Error: No pre-computed masks found in {mask_dir}.")
            print("Please run Stage 5 (Skyline Segmentation) first.")
            return 1

        # Create a dictionary for quick lookup of masks by their base name
        # e.g., 'binary_panorama_1010.png' -> 'panorama_1010'
        mask_map = {p.stem.replace("binary_", ""): p for p in mask_files}

        print(f"Found {len(image_files)} images and {len(mask_files)} masks.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_sub_dir = temp_path / "images"
            masks_sub_dir = temp_path / "masks"
            images_sub_dir.mkdir()
            masks_sub_dir.mkdir()

            # 1. Setup XML Structure
            root = Element("annotations")
            SubElement(root, "version").text = "1.1"
            meta = SubElement(root, "meta")
            task = SubElement(meta, "task")
            SubElement(task, "name").text = output_cvat_zip_path.stem
            labels_element = SubElement(task, "labels")
            sky_label = SubElement(labels_element, "label")
            SubElement(sky_label, "name").text = "sky"
            SubElement(sky_label, "color").text = "#0000FF"  # Blue

            mask_id_counter = 0

            # 2. Process Each Image and Match with its Mask
            for image_idx, image_file in enumerate(
                tqdm(image_files, desc="Packaging for CVAT")
            ):
                image_stem = image_file.stem
                if image_stem in mask_map:
                    mask_file_path = mask_map[image_stem]

                    # Copy original image
                    shutil.copy(image_file, images_sub_dir / image_file.name)

                    # Copy the pre-computed mask, renaming it sequentially
                    mask_id_counter += 1
                    new_mask_filename = f"{mask_id_counter}.png"
                    shutil.copy(mask_file_path, masks_sub_dir / new_mask_filename)

                    # Add image and mask info to XML
                    with Image.open(image_file) as img:
                        img_w, img_h = img.size

                    image_element = SubElement(
                        root,
                        "image",
                        id=str(image_idx),
                        name=f"images/{image_file.name}",
                        width=str(img_w),
                        height=str(img_h),
                    )
                    SubElement(
                        image_element,
                        "mask",
                        label="sky",
                        source="auto",  # Changed to 'auto' as it's from a model
                        occluded="0",
                        z_order="0",
                        subset=str(mask_id_counter),
                    )
                else:
                    print(
                        f"Warning: No matching mask found for image '{image_file.name}'. Skipping."
                    )

            # 3. Save XML and Create ZIP
            if mask_id_counter == 0:
                print(
                    "Error: No image-mask pairs were successfully processed. Aborting ZIP creation."
                )
                return 1

            xml_string = tostring(root, "utf-8")
            pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
            with open(temp_path / "annotations.xml", "w") as f:
                f.write(pretty_xml)

            print(f"Creating ZIP archive at {output_cvat_zip_path}...")
            shutil.make_archive(
                str(output_cvat_zip_path.with_suffix("")),  # make_archive adds .zip
                "zip",
                root_dir=temp_path,
            )

        print("\n" + "=" * 50)
        print("✅ CVAT Packaging Complete!")
        print(f"File saved to: {output_cvat_zip_path}")
        print("\nHow to use in CVAT:")
        print("  1. Create a new Task in CVAT.")
        print("  2. Add one label: 'sky'.")
        print(f"  3. In the task menu (Actions ...), select 'Upload annotations'.")
        print("  4. Choose 'CVAT 1.1' format and upload the ZIP file.")
        print("=" * 50 + "\n")
        return 0


class InteractiveSegmentationViewer:
    """
    Shows a side-by-side view with a semi-transparent binary sky mask.
    """

    def __init__(self, image_pil, class_id_map, id2label_map, sky_id):
        self.image_pil = image_pil
        self.class_id_map = class_id_map
        self.id2label = id2label_map
        self.sky_id = sky_id

        self.fig, self.ax = plt.subplots(
            1, 2, figsize=(20, 8), sharex=True, sharey=True
        )

        # LEFT: original + pseudo-color segmentation (works for binary or multiclass)
        self.ax[0].imshow(self.image_pil)
        self.ax[0].set_title("Segmentation (pseudo-color)")
        self.ax[0].axis("off")

        np.random.seed(42)
        max_id = int(np.max(class_id_map))
        cmap_size = max(
            max_id + 1, max(2, len(self.id2label) or 2)
        )  # at least 2 entries
        cmap_rgba = np.zeros((cmap_size, 4))
        for i in range(cmap_size):
            cmap_rgba[i, :3] = np.random.rand(3)
        cmap_rgba[:, 3] = 0.4

        seg_rgba = cmap_rgba[self.class_id_map]
        self.ax[0].imshow(seg_rgba)

        self.info_text = self.ax[0].text(
            0.01,
            0.99,
            "",
            transform=self.ax[0].transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 4},
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # RIGHT: explicit binary sky overlay
        self.ax[1].imshow(self.image_pil)
        self.ax[1].set_title("Binary Sky Mask (Overlay)")
        self.ax[1].axis("off")

        # If only {0,1} present, treat 1 as sky, else fall back to sky_id
        vals = np.unique(self.class_id_map)
        if set(vals.tolist()) <= {0, 1}:
            sky_mask = self.class_id_map.astype(bool)
        else:
            sky_mask = self.class_id_map == self.sky_id

        sky_overlay = np.zeros((*sky_mask.shape, 4), dtype=np.float32)
        sky_overlay[sky_mask] = [0, 0, 1, 0.4]
        self.ax[1].imshow(sky_overlay)

        plt.tight_layout()
        plt.show()

    def on_hover(self, event):
        if event.inaxes != self.ax[0]:
            return
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < self.class_id_map.shape[0] and 0 <= x < self.class_id_map.shape[1]:
            class_id = self.class_id_map[y, x]
            label_name = self.id2label.get(
                class_id, self.id2label.get(str(class_id), f"Unknown ID: {class_id}")
            )
            self.info_text.set_text(f"({x}, {y}): {label_name} (ID: {class_id})")
        else:
            self.info_text.set_text("")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    IMAGE_PATH = os.path.join(constants.PANORAMAS_DIR, "panorama_1010.jpg")

    if not os.path.exists(IMAGE_PATH):
        print(f"FATAL: Image not found at '{IMAGE_PATH}'")
        print("Please update the IMAGE_PATH variable in the script.")
    else:
        segmenter = SemanticSegmenter()
        if segmenter.model:
            segmenter.view_single_image(IMAGE_PATH)
        else:
            print("Could not run the viewer because the model failed to load.")

    print("\nScript finished.")
