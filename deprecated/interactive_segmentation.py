# interactive_inspector.py
import sys
import os
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg backend not available, falling back to default.")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants

IMAGE_PATH = os.path.join(constants.PANORAMAS_DIR, "panorama_1010.jpg") 
# Model from Hugging Face Hub
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

class TilingImageSegmenter:
    """
    Handles segmentation of large images using a sliding window (tiling) approach
    to preserve detail and manage memory.
    """
    def __init__(self, model_name):
        print("Initializing Tiling Image Segmenter...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
            self.id2label = self.model.config.id2label
            self.num_classes = self.model.config.num_labels
            print("Model and processor loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model '{model_name}'. Exception: {e}")
            self.model = None
            self.processor = None
            self.id2label = {}

    def get_pixel_level_segmentation_data(self, image_path, tile_size=640, overlap=64):
        """
        Segments a large image by processing it in overlapping tiles.
        
        Args:
            image_path (str): Path to the image file.
            tile_size (int): The size of the square tiles (should match model input).
            overlap (int): The overlap between adjacent tiles to reduce edge artifacts.

        Returns:
            dict: A dictionary containing the original image, the full-resolution
                class ID map, and the label mapping, or None on failure.
        """
        if not self.model or not os.path.exists(image_path):
            print(f"Error: Model not loaded or image not found at {image_path}")
            return None

        print(f"Processing {os.path.basename(image_path)} with a sliding window...")
        original_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(original_pil)
        img_h, img_w, _ = image_np.shape

        full_logits_map = np.zeros((img_h, img_w, self.num_classes), dtype=np.float32)
        
        stitch_counts = np.zeros((img_h, img_w, 1), dtype=np.float32)
        step = tile_size - overlap

        for y in range(0, img_h, step):
            for x in range(0, img_w, step):
                # Define tile boundaries
                y_end = min(y + tile_size, img_h)
                x_end = min(x + tile_size, img_w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = image_np[y_start:y_end, x_start:x_end, :]

                inputs = self.processor(images=tile, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits = outputs.logits # Shape: (1, num_classes, H/4, W/4)
                
                # Upsample logits to original tile size
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=tile.shape[:2],
                    mode="bilinear",
                    align_corners=False
                )
                
                # Permute dimensions to (H, W, num_classes) to match the numpy array
                logits_to_add = upsampled_logits.squeeze().permute(1, 2, 0).cpu().numpy()
                full_logits_map[y_start:y_end, x_start:x_end, :] += logits_to_add
                
                # Increment the stitch count
                stitch_counts[y_start:y_end, x_start:x_end, :] += 1
        
        # Avoid division by zero
        stitch_counts[stitch_counts == 0] = 1
        averaged_logits = full_logits_map / stitch_counts
        
        # Take the argmax along the last axis (the classes axis) to get the final ID map
        final_class_id_map = np.argmax(averaged_logits, axis=-1).astype(np.int32)
        
        print("Sliding window processing complete.")
        return {
            'original_image_pil': original_pil,
            'class_id_map': final_class_id_map,
            'id2label': self.id2label
        }

class InteractiveSegmentationViewer:
    """
    An interactive viewer that uses vectorized NumPy for fast colorization
    of the segmentation map.
    """
    def __init__(self, image_pil, class_id_map, id2label_map):
        self.image_pil = image_pil
        self.class_id_map = class_id_map 
        self.id2label = id2label_map     
        self.fig, self.ax = plt.subplots(figsize=(15, 8))

        max_id_in_map = int(np.max(self.class_id_map))
        num_labels_config = len(self.id2label)
        
        # Create a colormap for lookup, indexed by Class ID
        # The size is the max of what's in the map vs. what's in the config
        cmap_size = max(max_id_in_map, num_labels_config) + 1
        cmap_rgba = np.zeros((cmap_size, 4))
        
        np.random.seed(42)
        # Generate colors for every possible label from the model's config
        for str_id, label in self.id2label.items():
            int_id = int(str_id)
            if int_id < cmap_size:
                cmap_rgba[int_id, :3] = np.random.rand(3)
                cmap_rgba[int_id, 3] = 0.2  # Alpha for semi-transparency

        # Use the class ID map to instantly look up the color for every pixel
        segmented_rgba = cmap_rgba[self.class_id_map]
        
        self.ax.imshow(self.image_pil)
        self.ax.imshow(segmented_rgba)
        self.ax.set_title("Hover to Inspect Pixel Label | Sliding Window Segmentation")
        
        self.info_text = self.ax.text(0.01, 0.99, '', transform=self.ax.transAxes,
                                    verticalalignment='top', fontsize=10,
                                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 4})
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        plt.tight_layout()
        plt.show()

    def on_hover(self, event):
        # Displays the class label of the pixel under the mouse cursor
        if not event.inaxes:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < self.class_id_map.shape[0] and 0 <= x < self.class_id_map.shape[1]:
            class_id = self.class_id_map[y, x]
            label_name = self.id2label.get(class_id, f"Unknown ID: {class_id}")
            self.info_text.set_text(f"({x}, {y}): {label_name} (ID: {class_id})")
        else:
            self.info_text.set_text('')
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"FATAL: Image not found at '{IMAGE_PATH}'")
        print("Please update the IMAGE_PATH variable at the top of the script.")
    else:
        # 1. Initialize the segmenter with tiling capabilities
        segmenter = TilingImageSegmenter(model_name=MODEL_NAME)

        if segmenter.model:
            # 2. Get segmentation data using the sliding window method
            segmentation_data = segmenter.get_pixel_level_segmentation_data(IMAGE_PATH)

            if segmentation_data:
                print("Launching interactive viewer...")
                # 3. Launch the fast, vectorized viewer with the full-resolution data
                viewer = InteractiveSegmentationViewer(
                    segmentation_data['original_image_pil'],
                    segmentation_data['class_id_map'],
                    segmentation_data['id2label']
                )
            else:
                print("Failed to get segmentation data.")
    
    print("\nScript finished.")