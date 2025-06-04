# segmenter.py
import os
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np

import constants

def load_model_and_processor(model_name="nvidia/segformer-b0-finetuned-ade-512-512", device="cpu"):
    """Loads the SegFormer model and its processor."""
    print(f"Loading model: {model_name} on device: {device}")
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval() # Set to evaluation mode
        print("Model and processor loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading model/processor: {e}")
        print("Make sure you have an internet connection for the first download,")
        print("and that the model name is correct.")
        raise

def segment_sky(image_path, processor, model, device):
    """
    Segments an image to identify sky regions.

    Args:
        image_path (str): Path to the image file.
        processor (SegformerImageProcessor): The image processor.
        model (SegformerForSemanticSegmentation): The segmentation model.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        tuple: (original_image, full_segmentation_map, sky_mask)
                original_image (PIL.Image): The loaded original image.
                full_segmentation_map (np.array): The predicted class for each pixel.
                sky_mask (np.array): Boolean array, True where sky is detected.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        raise

    print(f"Processing image: {image_path} with size {image.size}")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # Shape: (batch_size, num_classes, height, width)

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    # Get segmentation map (class ID for each pixel)
    predicted_semantic_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Find the class ID for "sky"
    sky_class_id = None
    sky_label_name = "sky" # The label for sky in ADE20K
    
    # Check model's config for label mapping
    if hasattr(model.config, 'id2label'):
        for class_id, label in model.config.id2label.items():
            if label.lower() == sky_label_name:
                sky_class_id = class_id
                break
    
    if sky_class_id is None:
        print(f"Warning: Class '{sky_label_name}' not found in model labels. Available labels:")
        if hasattr(model.config, 'id2label'):
            print(model.config.id2label)

        print("Attempting to use default ADE20K sky ID (2) as a fallback.")
        sky_class_id = 2 # Common for ADE20K
        if sky_class_id not in model.config.id2label or model.config.id2label[sky_class_id].lower() != sky_label_name:
            raise ValueError(f"'{sky_label_name}' class ID could not be reliably determined. Please check model's label mapping.")


    print(f"Identified '{sky_label_name}' as class ID: {sky_class_id}")

    # Create binary sky mask
    sky_mask = (predicted_semantic_map == sky_class_id)

    return image, predicted_semantic_map, sky_mask

if __name__ == "__main__":
    IMAGE_PATH = os.path.join(constants.COLMAP_DIR, "images_cam1/cam1_0801.jpg")

    MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512" 

    # Check for image existence before proceeding
    try:
        with open(IMAGE_PATH, 'rb') as f:
            pass
    except FileNotFoundError:
        print(f"ERROR: Image file not found at '{IMAGE_PATH}'.")
        print("Please update the IMAGE_PATH variable in the script.")
        exit()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor, model = load_model_and_processor(MODEL_NAME, device)

    try:
        original_image, full_seg_map, sky_mask_np = segment_sky(IMAGE_PATH, processor, model, device)

        # --- Visualization ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Displaying the full segmentation map
        # For ADE20K, there are 150 classes, so direct visualization might be noisy.
        # `cmap='viridis'` or `cmap='jet'` can be used, but a custom colormap is better for many classes.
        # Here, `tab20` provides a good set of distinct colors for up to 20 classes.
        axs[1].imshow(full_seg_map, cmap='tab20', vmin=0, vmax=model.config.num_labels -1) # Use num_labels from model config
        axs[1].set_title("Full Semantic Segmentation")
        axs[1].axis('off')

        axs[2].imshow(sky_mask_np, cmap='gray') # Binary mask: sky is white, non-sky is black
        axs[2].set_title("Sky Mask (Sky=White)")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

        print("Sky mask shape:", sky_mask_np.shape)
        print(f"Sky pixels: {np.sum(sky_mask_np)}")
        print(f"Total pixels: {sky_mask_np.size}")
        print(f"Percentage of sky: {100 * np.sum(sky_mask_np) / sky_mask_np.size:.2f}%")

    except Exception as e:
        print(f"An error occurred during segmentation or visualization: {e}")