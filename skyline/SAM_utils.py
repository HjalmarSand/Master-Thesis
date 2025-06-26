import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor 
#HOW TO GET THE IMPORT: use "pip install -e ." in the segment-anything directory 
# to use the segment-anything library, or ensure it's in your PYTHONPATH. 

def setup_sam_predictor(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    """Initializes and returns a SAM predictor object."""
    print("Loading SAM model... This may take a moment.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")
    return predictor

# --- NEW FUNCTION TO CLEAN THE MASK ---
def _keep_largest_connected_component(mask):
    """
    Cleans a binary mask by keeping only the largest connected component.
    This is used to remove small, erroneous "sky" patches below the horizon.

    Args:
        mask (np.array): A 2D boolean mask from SAM.

    Returns:
        np.array: A cleaned 2D boolean mask.
    """
    # Find all connected components and their stats
    # The mask must be of type uint8 for this function
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    # If there are no foreground components, return the original mask
    if num_labels <= 1:
        return mask

    # The first label (0) is the background. We find the label of the largest
    # component by looking at the area stat (cv2.CC_STAT_AREA) for all other labels.
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new mask that is True only where the labels match the largest component
    cleaned_mask = (labels_im == largest_component_label)
    
    return cleaned_mask

def _extract_skyline_from_mask(mask):
    """(Internal) Extracts the skyline from a boolean sky mask."""
    height, width = mask.shape
    skyline = np.zeros(width, dtype=int)
    for x in range(width):
        sky_pixels_in_column = np.where(mask[:, x])[0]
        if sky_pixels_in_column.size > 0:
            # The skyline is the lowest 'sky' pixel in this column
            skyline[x] = np.max(sky_pixels_in_column)
        else:
            # If a column has no sky, default to the top of the image
            skyline[x] = 0
    return skyline

def extract_skyline_from_image(image_path, predictor):
    """
    Takes an image path and a SAM predictor, segments the sky, cleans the mask,
    and returns its 1D skyline vector.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(image)
    
    height, width, _ = image.shape
    # Your prompting strategy
    positive_points = np.array([
        [width // 4, height // 7], [width // 2, height // 7], [width * 3 // 4, height // 7],
    ])
    negative_points = np.array([
        [width // 2, height * 18 // 20], [width // 4, height * 15 // 20], [width * 3 // 4, height * 18 // 20],
    ])
    input_point = np.concatenate([positive_points, negative_points], axis=0)
    input_label = np.array([1, 1, 1, 0, 0, 0])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    # Get the raw mask from SAM
    raw_sky_mask = masks[0]

    # --- THE FIX: CLEAN THE MASK BEFORE EXTRACTING THE SKYLINE ---
    cleaned_sky_mask = _keep_largest_connected_component(raw_sky_mask)
    
    # Extract the skyline from the *cleaned* mask
    return _extract_skyline_from_mask(cleaned_sky_mask)

def compare_skylines_fft(query_skyline, db_skyline):
    """
    Compares two skylines using FFT-based normalized cross-correlation.
    Handles circular rotation to find the best possible match.
    Returns the max correlation score and the optimal rotation.
    """
    q = query_skyline.astype(float)
    d = db_skyline.astype(float)
    q = (q - np.mean(q)) / (np.std(q) + 1e-9)
    d = (d - np.mean(d)) / (np.std(d) + 1e-9)
    q_fft = np.fft.fft(q)
    d_fft = np.fft.fft(np.flip(d))
    correlation = np.fft.ifft(q_fft * d_fft)
    score = np.max(np.abs(correlation)) / len(q)
    rotation = np.argmax(np.abs(correlation))
    return score, rotation