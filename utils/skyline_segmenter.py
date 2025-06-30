import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

def get_sam_predictor():
    """Initializes and returns the SAM model predictor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing SAM predictor on device: {device}")
    
    model_type = "vit_b"
    # Ensure this checkpoint file is downloaded and accessible
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print("SAM predictor initialized successfully.")
        return predictor
    except FileNotFoundError:
        print(f"ERROR: SAM checkpoint file not found at '{sam_checkpoint}'.")
        print("Please download it from the official Segment-Anything repository and place it in your project root.")
        return None

def keep_largest_connected_component(mask):
    """
    Cleans a binary mask by keeping only the largest connected component.
    Handles the case where no components are found.
    """
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    # If there are no components (or only the background), return an empty mask
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    # Find the label of the largest component (excluding background at label 0)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a new mask that is True only for the largest component
    cleaned_mask = (labels_im == largest_component_label)
    
    return cleaned_mask

def generate_sky_mask(image_rgb, predictor):
    """
    Generates a clean, binary sky mask for a given image using a prompted SAM predictor.
    
    Args:
        image_rgb (np.ndarray): The input image in RGB format.
        predictor (SamPredictor): The initialized SAM predictor instance.
        
    Returns:
        np.ndarray: A boolean mask where True represents the sky, or None if prediction fails.
    """
    # 1. Set the image in the predictor
    predictor.set_image(image_rgb)
    
    # 2. Define positive and negative points to guide SAM
    height, width, _ = image_rgb.shape
    positive_points = np.array([
        [width // 4, height // 7], 
        [width // 2, height // 7], 
        [width * 3 // 4, height // 7],
    ])
    negative_points = np.array([
        [width // 2, height * 18 // 20], 
        [width // 4, height * 15 // 20], 
        [width * 3 // 4, height * 18 // 20],
    ])
    input_point = np.concatenate([positive_points, negative_points], axis=0)
    input_label = np.array([1, 1, 1, 0, 0, 0]) # 1 for positive, 0 for negative
    
    # 3. Predict the mask
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False, # We only want the single best mask
    )
    
    # 4. Clean the mask by keeping the largest component
    # The output mask from predictor is a boolean array
    cleaned_mask = keep_largest_connected_component(masks[0])
    
    return cleaned_mask

def segment_skyline_from_panoramas(panorama_dir, output_dir, predictor):
    """
    Processes all panoramas in a directory to extract and save binary skyline masks.
    
    Args:
        panorama_dir (str or Path): The directory containing panoramic images.
        output_dir (str or Path): The directory where masks will be saved.
        predictor (SamPredictor): An initialized SAM predictor.
    """
    panorama_path = Path(panorama_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    panorama_files = list(panorama_path.glob("*.jpg"))
    if not panorama_files:
        print(f"Error: No panoramas found in {panorama_path}")
        return
        
    for pano_file in tqdm(panorama_files, desc="Segmenting Skylines"):
        image_bgr = cv2.imread(str(pano_file))
        if image_bgr is None:
            tqdm.write(f"Warning: Could not load {pano_file.name}. Skipping.")
            continue
            
        # Convert to RGB for SAM
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Generate the clean sky mask
        sky_mask = generate_sky_mask(image_rgb, predictor)
        
        if sky_mask is not None:
            # Convert the boolean mask to a savable binary image (0s and 255s)
            final_mask_image = sky_mask.astype(np.uint8) * 255
            
            # Save the mask as a PNG to preserve quality
            output_filename = f"skymask_{pano_file.stem}.png"
            save_path = output_path / output_filename
            cv2.imwrite(str(save_path), final_mask_image)
        else:
            tqdm.write(f"Warning: Failed to generate a sky mask for {pano_file.name}.")

    print(f"Skyline segmentation complete. Masks saved to: {output_path}")
    return 0