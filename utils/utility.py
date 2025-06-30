import cv2
import os
import re

def get_image_dimensions_opencv(image_path):
    """
    Gets the dimensions (width, height) of an image using OpenCV.

    Args:
        image_path (str or pathlib.Path): The path to the image file.

    Returns:
        tuple: (width, height) in pixels, or None if an error occurs.
    """
    try:
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Error: OpenCV could not read the image at '{image_path}'.")
            return None

        
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        print(f"An unexpected error occurred with OpenCV while processing '{image_path}': {e}")
        return None

def get_original_numbered_file_name(file_name):
    """
    Robustly extracts the main 4-digit numerical identifier from a filename.
    
    This function is designed to work with various filename formats like:
    - '0801.jpg'
    - 'undistorted_cam1_0801_left.jpg'
    - 'image_0801.png'
    
    It finds the first sequence of exactly 4 digits and returns it as a
    standardized "number.jpg" filename.
    
    Args:
        file_name (str): The input filename.
        
    Returns:
        str: A simplified filename like "0801.jpg", or None if no 4-digit number is found.
    """
    # This regular expression looks for a sequence of exactly 4 digits.
    # \b matches a "word boundary", which prevents it from matching part of a larger number
    # (e.g., it won't match '1234' in '12345.jpg').
    # (\d{4}) is a capturing group for exactly 4 digits.
    match = re.search(r'\b(\d{4})\b', file_name)
    
    if match:
        # match.group(1) returns the captured 4-digit string, e.g., "0801"
        file_number = match.group(1)
        return f"{file_number}.jpg"
    else:
        # Fallback for cases where the number might not have word boundaries,
        # e.g., 'cam1_0801_left.jpg'
        match = re.search(r'(\d{4})', file_name)
        if match:
            file_number = match.group(1)
            return f"{file_number}.jpg"
            
    # If no 4-digit number is found after all attempts, return None.
    return None