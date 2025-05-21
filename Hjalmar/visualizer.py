import matplotlib.pyplot as plt
import cv2 # For loading images if paths are given
import numpy as np
from PIL import Image
from typing import Optional, Union

class Visualizer:

    def display_segmentation_results(self,
                                    image_source: Union[str, Image.Image, np.ndarray],
                                    sky_mask_np: Optional[np.ndarray] = None,
                                    skyline_vector: Optional[np.ndarray] = None,
                                    title_prefix: str = ""):
        
        plots_info = [] # List to store what to plot

        # Always have the original image
        plots_info.append("original")

        if sky_mask_np is not None:
            plots_info.append("mask")
        
        if skyline_vector is not None:
            plots_info.append("skyline")

        num_plots = len(plots_info)
        if num_plots == 0: # Should not happen due to "original" always being there
            return 

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False) # squeeze=False ensures axes is always 2D
        axes = axes.flatten() # Flatten to 1D array for easy indexing
        
        current_ax_idx = 0

        # --- Load Original Image ---
        # (Keep your existing image loading logic here)
        if isinstance(image_source, str):
            try:
                original_image_pil = Image.open(image_source).convert("RGB")
                original_image_np = np.array(original_image_pil)
            except FileNotFoundError:
                print(f"Error: Image not found at {image_source} for visualization.")
                plt.close(fig)
                return
        elif isinstance(image_source, Image.Image):
            original_image_pil = image_source.convert("RGB")
            original_image_np = np.array(original_image_pil)
        elif isinstance(image_source, np.ndarray):
            original_image_np = image_source 
        else:
            print("Error: Invalid image_source type for visualization.")
            plt.close(fig)
            return

        if "original" in plots_info:
            axes[current_ax_idx].imshow(original_image_np)
            axes[current_ax_idx].set_title(f"{title_prefix}Original Image")
            axes[current_ax_idx].axis('off')
            current_ax_idx += 1
        
        if "mask" in plots_info:
            if sky_mask_np is not None: # Double check, though it should be if "mask" is in plots_info
                axes[current_ax_idx].imshow(sky_mask_np * 255, cmap='gray')
                axes[current_ax_idx].set_title(f"{title_prefix}Sky Mask")
                axes[current_ax_idx].axis('off')
                current_ax_idx += 1
            else: # Should not happen if logic is correct, but good for robustness
                print("Warning: 'mask' in plots_info but sky_mask_np is None.")
                if current_ax_idx < len(axes): # Remove the unused axis slot if pre-allocated
                    fig.delaxes(axes[current_ax_idx])


        if "skyline" in plots_info:
            if skyline_vector is not None: # Double check
                overlay_image = original_image_np.copy() # Use a copy of the original for overlay
                
                axes[current_ax_idx].imshow(overlay_image) # Show image first
                
                # Overlay the skyline points/line
                x_coords = np.arange(len(skyline_vector))
                
                # Filter points that are within image bounds before plotting
                valid_indices = (skyline_vector >= 0) & (skyline_vector < overlay_image.shape[0]) & \
                                (x_coords >= 0) & (x_coords < overlay_image.shape[1])
                
                axes[current_ax_idx].plot(x_coords[valid_indices], skyline_vector[valid_indices], 'r-', linewidth=1)
                axes[current_ax_idx].scatter(x_coords[valid_indices], skyline_vector[valid_indices], c='red', s=5)


                axes[current_ax_idx].set_title(f"{title_prefix}Skyline Overlay")
                axes[current_ax_idx].axis('off')
                axes[current_ax_idx].set_ylim(overlay_image.shape[0], 0) 
                axes[current_ax_idx].set_xlim(0, overlay_image.shape[1])
                current_ax_idx += 1
            else: # Should not happen
                print("Warning: 'skyline' in plots_info but skyline_vector is None.")
                if current_ax_idx < len(axes):
                    fig.delaxes(axes[current_ax_idx])


        for i in range(current_ax_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()