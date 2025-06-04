import cv2
import os

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