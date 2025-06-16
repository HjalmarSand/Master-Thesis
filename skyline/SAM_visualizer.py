import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# HOW TO ACSESS SEGMENT ANYTHING: use "pip install -e ." in the segment-anything directory or ensure it's in your PYTHONPATH. 

# --- HELPER FUNCTIONS ---

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.5])):
    """Helper function to display a mask on a subplot axis."""
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Helper function to display points on a subplot axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def extract_skyline(mask):
    """Extracts the skyline from a boolean sky mask."""
    height, width = mask.shape
    skyline = np.zeros(width, dtype=int)
    for x in range(width):
        sky_pixels_in_column = np.where(mask[:, x])[0]
        if sky_pixels_in_column.size > 0:
            skyline[x] = np.max(sky_pixels_in_column)
        else:
            skyline[x] = 0
    return skyline

# --- NEW FUNCTION TO CLEAN THE MASK ---
def keep_largest_connected_component(mask):
    """
    Cleans a binary mask by keeping only the largest connected component.
    """
    # The mask must be of type uint8 for the OpenCV function
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return mask

    # Find the label of the largest component (excluding background at label 0)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a new mask that is True only for the largest component
    cleaned_mask = (labels_im == largest_component_label)
    
    return cleaned_mask

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image_path = 'images/cam1_undistorted_images/undistorted_cam1_1393.png' 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# --- PROMPT STRATEGY USING POINTS ---
height, width, _ = image.shape
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
input_label = np.array([1, 1, 1, 0, 0, 0])

# --- PREDICTION ---
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
# 1. Get the raw mask and skyline
raw_sky_mask = masks[0]
raw_skyline = extract_skyline(raw_sky_mask)

# 2. Apply the cleaning function to get the cleaned mask and skyline
cleaned_sky_mask = keep_largest_connected_component(raw_sky_mask)
cleaned_skyline = extract_skyline(cleaned_sky_mask)


# --- VISUALIZATION: SIDE-BY-SIDE COMPARISON ---
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# --- LEFT PLOT: BEFORE CLEANING ---
ax1 = axes[0]
ax1.imshow(image)
show_mask(raw_sky_mask, ax1, color=np.array([255/255, 69/255, 0/255, 0.5])) 
show_points(input_point, input_label, ax1)
x_coords = np.arange(width)
ax1.plot(x_coords, raw_skyline, color='yellow', linewidth=3, label='Raw Skyline')
ax1.set_title("Before Cleaning (Raw SAM Output)", fontsize=16)
ax1.axis('off')
ax1.legend()

# --- RIGHT PLOT: AFTER CLEANING ---
ax2 = axes[1]
ax2.imshow(image)
show_mask(cleaned_sky_mask, ax2, color=np.array([30/255, 144/255, 255/255, 0.5])) 
show_points(input_point, input_label, ax2)
ax2.plot(x_coords, cleaned_skyline, color='cyan', linewidth=3, label='Cleaned Skyline')
ax2.set_title("After Cleaning (Largest Component Kept)", fontsize=16)
ax2.axis('off')
ax2.legend()

plt.suptitle("Skyline Artifact Removal: Before vs. After", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()