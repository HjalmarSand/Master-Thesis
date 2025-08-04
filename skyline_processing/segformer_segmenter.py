import argparse
import os
import sys
from pathlib import Path

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

import constants


class SemanticSegmenter:
    """
    Handles semantic segmentation of images using a tiling approach.
    Can process a directory of images or launch an interactive viewer for a single image.
    """

    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640"):
        print("Initializing Semantic Segmenter...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.id2label = {}
        self.num_classes = 0
        self.sky_id = None

        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name
            ).to(self.device)
            self.id2label = self.model.config.id2label
            self.num_classes = self.model.config.num_labels
            self._find_sky_id()
            print(f"Using device: {self.device}")
            print("Model and processor loaded successfully.")
        except Exception as e:
            print(
                f"CRITICAL ERROR: Failed to load model '{self.model_name}'. Exception: {e}"
            )

    def _find_sky_id(self):
        """Helper method to find the integer ID for the 'sky' label."""
        for an_id, label in self.id2label.items():
            if "sky" in label.lower():
                self.sky_id = int(an_id)
                print(f"Found sky label '{label}' with ID: {self.sky_id}")
                return
        print("Warning: Could not find a 'sky' label in the model's configuration.")

    def _segment_single_image(self, image_path, tile_size=640, overlap=64):
        """
        Internal method to segment a single large image by processing it in overlapping tiles.
        """
        if not self.model or not os.path.exists(image_path):
            print(f"Error: Model not loaded or image not found at {image_path}")
            return None

        original_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(original_pil)
        img_h, img_w, _ = image_np.shape

        full_logits_map = np.zeros((img_h, img_w, self.num_classes), dtype=np.float32)
        stitch_counts = np.zeros((img_h, img_w, 1), dtype=np.float32)
        step = tile_size - overlap

        pbar = tqdm(
            total=((img_h - overlap) // step) * ((img_w - overlap) // step),
            desc=f"Segmenting {os.path.basename(image_path)}",
            leave=False,
        )

        for y in range(0, img_h, step):
            for x in range(0, img_w, step):
                y_end = min(y + tile_size, img_h)
                x_end = min(x + tile_size, img_w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                tile = image_np[y_start:y_end, x_start:x_end, :]

                inputs = self.processor(images=tile, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)

                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=tile.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )

                logits_to_add = (
                    upsampled_logits.squeeze().permute(1, 2, 0).cpu().numpy()
                )
                full_logits_map[y_start:y_end, x_start:x_end, :] += logits_to_add
                stitch_counts[y_start:y_end, x_start:x_end, :] += 1
                pbar.update(1)

        pbar.close()
        stitch_counts[stitch_counts == 0] = 1
        averaged_logits = full_logits_map / stitch_counts
        final_class_id_map = np.argmax(averaged_logits, axis=-1)

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

    def process_directory(self, input_dir, output_dir_black_white, output_dir_overlay):
        """
        Processes all images, saving binary masks and/or overlay images to the
        specified directories. If an output directory is None, that step is skipped.
        """
        if not self.model:
            print("Model not loaded, cannot process directory.")
            return 1

        # Only create directories if they are not None
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

            # Determine if we need to generate each output type by checking if the
            # directory was provided and if the file doesn't already exist.
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

            # If all necessary files already exist, skip the expensive segmentation
            if not do_black_white and not do_overlay:
                continue

            segmentation_data = self._segment_single_image(str(image_file))

            if segmentation_data:
                class_map = segmentation_data["class_id_map"]
                original_pil = segmentation_data["original_image_pil"]

                if self.sky_id is not None:
                    sky_mask_bool = class_map == self.sky_id

                    if do_black_white:
                        output_binary_bw_file = (
                            output_dir_black_white / f"binary_{base_name}.png"
                        )
                        sky_mask_bw_img = Image.fromarray(
                            sky_mask_bool.astype(np.uint8) * 255, "L"
                        )
                        sky_mask_bw_img.save(output_binary_bw_file)

                    if do_overlay:
                        output_binary_overlay_file = (
                            output_dir_overlay / f"binary_overlay_{base_name}.png"
                        )
                        overlay = Image.new("RGBA", original_pil.size, (0, 0, 255, 0))
                        overlay_pixels = np.array(overlay)
                        overlay_pixels[sky_mask_bool] = [0, 0, 255, 102]
                        overlay = Image.fromarray(overlay_pixels, "RGBA")

                        original_rgba = original_pil.convert("RGBA")
                        overlaid_image = Image.alpha_composite(original_rgba, overlay)
                        overlaid_image.save(output_binary_overlay_file)

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
        Args:
            image_dir (Path): Directory containing the original panorama images.
            mask_dir (Path): Directory containing the pre-computed binary masks
                            from Stage 5 (e.g., 'binary_panorama_1010.png').
            output_cvat_zip_path (Path): The full path for the output ZIP file.
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

            # --- 1. Setup XML Structure ---
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

            # --- 2. Process Each Image and Match with its Mask ---
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

            # --- 3. Save XML and Create ZIP ---
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

        # --- Plot 1: Full Semantic Segmentation (Left) ---
        self.ax[0].imshow(self.image_pil)
        self.ax[0].set_title("Full Semantic Segmentation")
        self.ax[0].axis("off")

        np.random.seed(42)
        max_id = int(np.max(class_id_map))
        cmap_size = max(max_id + 1, len(self.id2label))
        cmap_rgba = np.zeros((cmap_size, 4))
        for i in range(cmap_size):
            cmap_rgba[i, :3] = np.random.rand(3)
        cmap_rgba[:, 3] = 0.4

        segmented_rgba = cmap_rgba[self.class_id_map]
        self.ax[0].imshow(segmented_rgba)

        self.info_text = self.ax[0].text(
            0.01,
            0.99,
            "",
            transform=self.ax[0].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 4},
        )

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        self.ax[1].imshow(self.image_pil)
        self.ax[1].set_title("Binary Sky Mask (Overlay)")
        self.ax[1].axis("off")

        if self.sky_id is not None:
            sky_mask = self.class_id_map == self.sky_id
            sky_overlay = np.zeros((*sky_mask.shape, 4), dtype=np.float32)
            sky_overlay[sky_mask] = [0, 0, 1, 0.4]  # Blue color with 40% opacity
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
