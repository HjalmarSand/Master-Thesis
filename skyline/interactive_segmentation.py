# interactive_inspector.py
import matplotlib

import constants

try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg backend not available. Falling back.")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from PIL import Image 
from Hjalmar.skyline.image_segmenter import ImageSegmenter 

SEGMENTATION_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_SKY_LABEL = "sky"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_TO_INSPECT = os.path.join(BASE_DIR, "dataset1","split_images", "0802_back.jpg") 

class InteractiveSegmentationViewer:
    def __init__(self, image_pil, class_id_map, id2label_map):
        self.image_pil = image_pil
        self.class_id_map = class_id_map 
        self.id2label = id2label_map     

        self.fig, self.ax = plt.subplots(figsize=(10, 8)) 

        num_classes = len(self.id2label)
        # Generate distinct colors
        np.random.seed(42) # for reproducibility

        raw_colors = [tuple(np.random.rand(3)) for _ in range(num_classes)]
        
        unique_ids_in_map = np.unique(self.class_id_map)
        max_id_in_map = np.max(unique_ids_in_map) if unique_ids_in_map.size > 0 else -1
        
        cmap_colors = [(0,0,0,0)] * (max_id_in_map + 2) # +2 for safety
        
        for str_id, label in self.id2label.items():
            try:
                int_id = int(str_id)
                if 0 <= int_id < len(cmap_colors): # Ensure within bounds
                    color_idx = hash(label) % len(raw_colors)
                    cmap_colors[int_id] = (*raw_colors[color_idx], 0.5) # RGB + Alpha (0.5 for semi-transparency)
            except ValueError:
                print(f"Warning: Could not convert id '{str_id}' to int for colormap.")

        # Handle the -1 "unknown/background" ID specifically if present
        if -1 in unique_ids_in_map:
            pass 

        segmented_rgba = np.zeros((self.class_id_map.shape[0], self.class_id_map.shape[1], 4), dtype=float)

        for r in range(self.class_id_map.shape[0]):
            for c in range(self.class_id_map.shape[1]):
                class_id = self.class_id_map[r, c]
                if 0 <= class_id < len(cmap_colors):
                    segmented_rgba[r, c, :] = cmap_colors[class_id]

        # Display original image
        self.ax.imshow(self.image_pil)
        # Overlay the colorized segmentation map
        self.seg_display = self.ax.imshow(segmented_rgba) # This will blend based on alpha

        self.ax.set_title("Hover to see class label | Colorized Segments")
        self.info_text = self.ax.text(0.01, 0.95, '', transform=self.ax.transAxes,
                                    verticalalignment='top',
                                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3})
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        plt.tight_layout()
        plt.show()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)

            if 0 <= y < self.class_id_map.shape[0] and 0 <= x < self.class_id_map.shape[1]:
                class_id_val = self.class_id_map[y, x] # This is an integer ID
                
                class_id_str_key = str(class_id_val)
                
                label_name = self.id2label.get(class_id_str_key, f"Unknown ID: {class_id_val}")
                
                # FIX FOR "Unknown ID" for every pixel:
                if label_name.startswith("Unknown ID") and class_id_val in self.id2label:
                    label_name = self.id2label.get(class_id_val, f"Unknown ID (int key fallback): {class_id_val}")


                self.info_text.set_text(f"Pixel ({x},{y}): {label_name} (ID: {class_id_val})")
                self.fig.canvas.draw_idle()
            else:
                self.info_text.set_text('')
                self.fig.canvas.draw_idle()
        else:
            self.info_text.set_text('')
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    print("Initializing Image Segmenter for interactive inspection...")
    segmenter = ImageSegmenter(model_name=SEGMENTATION_MODEL_NAME, sky_label_name=DEFAULT_SKY_LABEL)

    if segmenter.segmenter is None:
        print("CRITICAL ERROR: Segmentation model failed to load.")
    else:
        if not os.path.exists(IMAGE_TO_INSPECT):
            print(f"ERROR: Image to inspect not found at {IMAGE_TO_INSPECT}")
            print(f"Please update IMAGE_TO_INSPECT in interactive_inspector.py")
        else:
            print(f"Processing image: {IMAGE_TO_INSPECT}")
            segmentation_data = segmenter.get_pixel_level_segmentation_data(IMAGE_TO_INSPECT)

            if segmentation_data:
                print("\n--- Debugging id2label mapping ---")
                print(f"Type of id2label: {type(segmentation_data['id2label'])}")
                if segmentation_data['id2label']:
                    first_key = next(iter(segmentation_data['id2label']))
                    print(f"Type of first key in id2label: {type(first_key)}")
                    print(f"First few entries (max 5):")
                    for i, (k, v) in enumerate(segmentation_data['id2label'].items()):
                        print(f"  Key: {k} (type: {type(k)}), Value: {v}")
                        if i >= 4: break
                print("---------------------------------\n")


                print("Launching interactive viewer...")
                viewer = InteractiveSegmentationViewer(
                    segmentation_data['original_image_pil'],
                    segmentation_data['class_id_map'],
                    segmentation_data['id2label']
                )
            else:
                print("Failed to get segmentation data for interactive viewer.")
    print("\n--- Interactive inspection script finished (or viewer closed) ---")