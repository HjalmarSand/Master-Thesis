import numpy as np
from typing import Optional

class SkylineExtractor:
    """
    Extracts the skyline profile from a binary sky mask.
    The skyline is defined as the y-coordinate of the lowest sky pixel in each column.
    """
    def __init__(self):
        """
        Initializes the SkylineExtractor.
        (Currently no specific configuration needed for initialization)
        """
        pass

    def extract_skyline(self, sky_mask_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the skyline from a given binary sky mask.

        Args:
            sky_mask_np (np.ndarray): A 2D NumPy array (H, W) where 1 is sky, 0 is non-sky.

        Returns:
            Optional[np.ndarray]: A 1D NumPy array of length W representing the y-coordinate 
                                of the skyline for each column. Returns None if the mask is invalid.
        """
        if sky_mask_np is None or sky_mask_np.ndim != 2 or sky_mask_np.size == 0:
            print("Error: Invalid sky mask provided for skyline extraction.")
            return None
        
        height, width = sky_mask_np.shape
        skyline_profile = np.zeros(width, dtype=np.int32) 

        for x_col in range(width):
            # Find indices of sky pixels (value == 1) in the current column
            sky_pixel_indices_in_column = np.where(sky_mask_np[:, x_col] == 1)[0]
            
            if sky_pixel_indices_in_column.size > 0:
                # The skyline is the y-coordinate of the lowest sky pixel in this column
                skyline_profile[x_col] = np.max(sky_pixel_indices_in_column)
            else:
                # No sky pixels in this column (e.g., a tall cliff fills the view).
                skyline_profile[x_col] = 0 
                
        return skyline_profile