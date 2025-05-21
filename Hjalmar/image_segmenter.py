# segmenter.py
from PIL import Image
import numpy as np
from transformers import pipeline, Pipeline
from typing import Optional, Union

class ImageSegmenter:
    """
    Handles semantic segmentation of images, specifically to identify the sky.
    """
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512", sky_label_name: str = "sky"):
        """
        Initializes the ImageSegmenter with a specified Hugging Face model.

        Args:
            model_name (str): The name of the pre-trained segmentation model 
                              from Hugging Face Hub.
            sky_label_name (str): The label name used by the model to identify 'sky'.
                                  This might need to be verified against the model's config.
        """
        print(f"Loading segmentation model: {model_name}...")
        try:
            self.segmenter: Pipeline = pipeline("image-segmentation", model=model_name)
            print("Segmentation model loaded successfully.")
        except Exception as e:
            self.segmenter = None

        self.sky_label_name = sky_label_name.lower() # For case-insensitive matching

    def get_sky_mask(self, image_source: Union[str, Image.Image]) -> Optional[np.ndarray]:
        """
        Generates a binary mask identifying sky pixels in the given image.

        Args:
            image_source (Union[str, Image.Image]): Path to the image file or a PIL Image object.

        Returns:
            Optional[np.ndarray]: A 2D NumPy array (H, W) where 1 represents sky 
                                  and 0 represents non-sky. Returns None if segmentation fails
                                  or the model is not loaded.
        """
        if self.segmenter is None:
            print("Error: Segmentation model not loaded. Cannot process image.")
            return None

        try:
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, Image.Image):
                image = image_source.convert("RGB")
            else:
                raise ValueError("image_source must be a file path (str) or a PIL.Image object.")
        except FileNotFoundError:
            print(f"Error: Image not found at {image_source}")
            return None
        except Exception as e:
            print(f"Error loading image {image_source}: {e}")
            return None
        
        print(f"Segmenting image...")
        try:
            segmentation_results = self.segmenter(image)
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None

        sky_mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        found_sky_segment = False

        for segment_info in segmentation_results:
            if segment_info['label'].lower() == self.sky_label_name:
                mask_pil = segment_info['mask'] # Assuming the mask is a PIL Image
                
                # Ensure mask_pil is the same size as original image
                if mask_pil.size != image.size:
                    mask_pil = mask_pil.resize(image.size, Image.NEAREST)

                current_sky_segment = np.array(mask_pil, dtype=np.uint8) 
                sky_mask_np[current_sky_segment > 128] = 1 
                found_sky_segment = True
        
        if not found_sky_segment:
            print(f"Warning: Sky label '{self.sky_label_name}' not found in segmentation results.")
            return np.zeros((image.height, image.width), dtype=np.uint8)
            
        return sky_mask_np