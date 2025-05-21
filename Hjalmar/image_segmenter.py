# ImageSegmenter.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import pipeline, Pipeline 
from typing import Optional, Union, List, Dict, Any

class ImageSegmenter:
    """
    Handles semantic segmentation of images.
    Allows for sky mask generation (using a single sky_label_name for now)
    and full segment inspection of all classes.
    """
    def __init__(self,
                model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                sky_label_name: str = "sky"): 
        """
        Initializes the ImageSegmenter with a specified Hugging Face model.
        """
        print(f"Loading segmentation model: {model_name}...")
        self.segmenter: Optional[Pipeline] = None
        try:
            self.segmenter = pipeline("image-segmentation", model=model_name)
            print("Segmentation model loaded successfully.")
        except ImportError as ie:
            print("Ensure a deep learning backend like PyTorch is installed.")
        except Exception as e:
            print(f"Unexpected error loading model: {e}")

        self.sky_label_name = sky_label_name.lower().strip() # For get_sky_mask

    def _load_image(self, image_source: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Helper to load and convert image to RGB PIL Image."""
        try:
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, Image.Image):
                image = image_source.convert("RGB")
            else:
                raise ValueError("image_source must be a file path (str) or a PIL.Image object.")
            return image
        except FileNotFoundError:
            print(f"Error: Image not found at {image_source}")
            return None
        except Exception as e:
            print(f"Error loading image {image_source}: {e}")
            return None

    def get_sky_mask(self, image_source: Union[str, Image.Image]) -> Optional[np.ndarray]:
        """
        Generates a binary mask identifying sky pixels (using self.sky_label_name).
        (This is your original method, mostly unchanged)
        """
        if self.segmenter is None:
            print("Error: Segmentation model not loaded. Cannot process image for sky mask.")
            return None

        image = self._load_image(image_source)
        if image is None:
            return None

        try:
            segmentation_results = self.segmenter(image)
        except Exception as e:
            print(f"Error during segmentation for sky mask: {e}")
            return None

        if not segmentation_results:
            print(f"Warning: Segmentation returned no results for {image_source if isinstance(image_source, str) else 'PIL Image'}.")
            return np.zeros((image.height, image.width), dtype=np.uint8)

        sky_mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        found_sky_segment = False

        for segment_info in segmentation_results:
            if segment_info['label'].lower().strip() == self.sky_label_name:
                mask_pil = segment_info['mask']
                if mask_pil.size != image.size:
                    mask_pil = mask_pil.resize(image.size, Image.NEAREST)
                current_sky_segment_np = np.array(mask_pil, dtype=np.uint8)
                sky_mask_np[current_sky_segment_np > 128] = 1
                found_sky_segment = True

        if not found_sky_segment:
            img_src_repr = image_source if isinstance(image_source, str) else "PIL image"
            print(f"Warning: Sky label '{self.sky_label_name}' not found in segmentation results for {img_src_repr}.")
            return np.zeros((image.height, image.width), dtype=np.uint8)

        return sky_mask_np

    def get_all_segments_from_image(self, image_source: Union[str, Image.Image]) -> Optional[List[Dict[str, Any]]]:
        """
        Performs segmentation and returns all raw identified segments with their info.

        Args:
            image_source (Union[str, Image.Image]): Path to the image or a PIL Image.

        Returns:
            Optional[List[Dict[str, Any]]]: List of segment dictionaries (label, score, mask),
                                            or None on failure.
        """
        if self.segmenter is None:
            print("Error: Segmentation model not loaded. Cannot get all segments.")
            return None

        image = self._load_image(image_source)
        if image is None:
            return None

        print(f"Performing full segmentation for inspection on image...")
        try:
            all_segments = self.segmenter(image) 
            if all_segments:
                unique_labels = sorted(list(set([s['label'] for s in all_segments])))
                print(f"Found {len(all_segments)} segments. Unique labels: {unique_labels}")
            else:
                print(f"No segments found by the model for {image_source if isinstance(image_source, str) else 'PIL Image'}.")
            return all_segments
        except Exception as e:
            print(f"Error during full segmentation for inspection: {e}")
            return None

    def create_segmentation_visualization(self,
                                        image_source: Union[str, Image.Image],
                                        all_segments_data: Optional[List[Dict[str, Any]]] = None) -> Optional[Image.Image]:
        """
        Creates a PIL Image visualizing all detected segments with colored overlays and labels.

        Args:
            image_source (Union[str, Image.Image]): Original image path or PIL Image.
            all_segments_data (Optional[List[Dict[str, Any]]]): Pre-computed segments.
                                                            If None, runs segmentation.
        Returns:
            Optional[Image.Image]: Visualized PIL Image, or None on failure.
        """
        original_image = self._load_image(image_source)
        if original_image is None:
            return None

        if all_segments_data is None:
            all_segments_data = self.get_all_segments_from_image(original_image) 
            if all_segments_data is None:
                return None

        if not all_segments_data:
            print("No segments to visualize.")
            return original_image 

        vis_image = original_image.copy().convert("RGBA")
        draw = ImageDraw.Draw(vis_image)

        unique_labels = sorted(list(set([s['label'] for s in all_segments_data])))
        np.random.seed(42) 
        colors = [tuple(np.random.randint(50, 206, 3)) for _ in unique_labels]
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
                print("Warning: Specific fonts not found, using Pillow's default. Text might be basic.")

        for segment_info in reversed(all_segments_data): # Draw smaller segments on top
            label = segment_info['label']
            mask_pil = segment_info['mask']

            if mask_pil.size != original_image.size:
                mask_pil = mask_pil.resize(original_image.size, Image.NEAREST)

            mask_np_bool = np.array(mask_pil) > 128

            color = label_to_color.get(label, (128, 128, 128)) 

            segment_color_layer_np = np.zeros((original_image.height, original_image.width, 4), dtype=np.uint8)
            segment_color_layer_np[mask_np_bool, :3] = color
            segment_color_layer_np[mask_np_bool, 3] = 100 

            segment_color_layer_pil = Image.fromarray(segment_color_layer_np, 'RGBA')
            vis_image.alpha_composite(segment_color_layer_pil)

            if np.any(mask_np_bool):
                y_coords, x_coords = np.where(mask_np_bool)
                if y_coords.size > 0: 
                    text_x, text_y = int(x_coords.mean()), int(y_coords.mean())
                    draw.text((text_x, text_y), label, fill=(0,0,0), font=font, anchor="mm")

        vis_image = vis_image.convert("RGB")
        return vis_image

    def get_pixel_level_segmentation_data(self, image_source: Union[str, Image.Image]) -> Optional[Dict[str, Any]]:
        """
        Performs segmentation and returns data needed for pixel-level inspection:
        the original image, a per-pixel class ID map, and the id2label mapping.

        Args:
            image_source (Union[str, Image.Image]): Path to the image or a PIL Image.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing:
                'original_image_pil': The loaded PIL Image.
                'class_id_map': A 2D NumPy array (H, W) where each value is a class ID.
                'id2label': The model's dictionary mapping class IDs to label names.
                Returns None on failure.
        """
        if self.segmenter is None:
            print("Error: Segmentation model not loaded.")
            return None
        if not hasattr(self.segmenter.model.config, 'id2label'):
            print("Error: Model config does not have id2label mapping.")
            return None


        original_image_pil = self._load_image(image_source)
        if original_image_pil is None:
            return None

        print(f"Performing segmentation for pixel-level data on image...")
        try:
            # To get the raw class ID map, we might need to access lower-level outputs
            # The standard pipeline(image) returns processed segments.
            # We need to invoke the model more directly or see if there's a pipeline option.

            # Option 1: If pipeline has a way to return raw logits/maps (less common for high-level)
            # outputs = self.segmenter(original_image_pil, output_segmentation_map=True) # Fictional arg

            # Option 2: A more robust way is to use the processor and model directly
            # This is more involved than the simple pipeline call.
            # For SegFormer, the pipeline usually does this internally:
            # 1. Preprocess image with `self.segmenter.image_processor`
            # 2. Pass inputs to `self.segmenter.model`
            # 3. Post-process logits (argmax) to get class IDs

            # Let's try to leverage the existing pipeline's structure if possible,
            # or reconstruct the class ID map from the segments.
            # Reconstructing from segments can be lossy if segments overlap or don't cover all pixels.

            # A practical approach with the current `pipeline` for image-segmentation:
            # The `pipeline` returns a list of dictionaries, each with a 'mask' and 'label'.
            # We can reconstruct a class_id_map from this.
            # First, get the model's label2id mapping to assign correct IDs.
            
            if not hasattr(self.segmenter.model.config, 'label2id'):
                 print("Error: Model config does not have label2id mapping.")
                 # Fallback: create a temporary one if we have id2label
                 if hasattr(self.segmenter.model.config, 'id2label'):
                     label2id = {v: k for k, v in self.segmenter.model.config.id2label.items()}
                 else:
                     return None # Cannot proceed
            else:
                label2id = self.segmenter.model.config.label2id


            all_segments = self.segmenter(original_image_pil) # This is List[Dict]
            if not all_segments:
                print("No segments found by the model.")
                # Return an empty map or handle as an error
                class_id_map_np = np.full((original_image_pil.height, original_image_pil.width), -1, dtype=int) # -1 for unknown/bg
            else:
                class_id_map_np = np.full((original_image_pil.height, original_image_pil.width), -1, dtype=int) # Default to a background/unknown ID

                # Iterate through segments. For models like Segformer, masks might overlap,
                # and the pipeline already resolved this. Usually, the output masks are binary
                # for the predicted class. We need to fill our map.
                # The pipeline's output is usually a list of masks for *distinct* regions.
                for segment_info in all_segments:
                    label = segment_info['label']
                    class_id_str = label2id.get(label) # label2id values are string IDs for ADE20k Segformer
                    if class_id_str is None:
                        print(f"Warning: Label '{label}' not in model's label2id mapping. Skipping.")
                        continue
                    class_id = int(class_id_str) # Convert string ID to int

                    mask_pil = segment_info['mask']
                    if mask_pil.size != original_image_pil.size:
                        mask_pil = mask_pil.resize(original_image_pil.size, Image.NEAREST)
                    
                    mask_np_bool = np.array(mask_pil) > 128 # Get boolean mask for this segment
                    class_id_map_np[mask_np_bool] = class_id # Assign class ID to pixels in this mask

            return {
                'original_image_pil': original_image_pil,
                'class_id_map': class_id_map_np,
                'id2label': self.segmenter.model.config.id2label
            }

        except Exception as e:
            print(f"Error during pixel-level segmentation data retrieval: {e}")
            return None
