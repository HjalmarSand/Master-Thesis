# main_inspector.py
# gets a segmentation of a image and saves it as a jpg
import os
from PIL import Image 
from image_segmenter import ImageSegmenter 

SEGMENTATION_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_SKY_LABEL = "sky" #Altough not used in this script, necessary for ImageSegmenter initialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_TO_INSPECT = os.path.join(BASE_DIR, "dataset1", "split_images", "0801_back.jpg") 
OUTPUT_VISUALIZATION_PATH = os.path.join(BASE_DIR, "full_segmentation_visualization.png")

def run_inspection(image_path: str, output_path: str, segmenter_instance: ImageSegmenter):
    """
    Performs segmentation inspection on a single image and saves the visualization.
    """
    print(f"\n--- Inspecting All Segments for: {image_path} ---")

    if not os.path.exists(image_path):
        print(f"ERROR: Image to inspect not found at {image_path}")
        return

    all_segments = segmenter_instance.get_all_segments_from_image(image_path)

    if all_segments:
        visualization_pil_image = segmenter_instance.create_segmentation_visualization(
            image_path,
            all_segments_data=all_segments
        )

        if visualization_pil_image:
            try:
                visualization_pil_image.save(output_path)
                print(f"Success: Segmentation visualization saved to: {output_path}")
            except Exception as e:
                print(f"ERROR: Could not save or show visualization: {e}")
        else:
            print("Failed to create visualization image.")
    else:
        print("No segments found or error during segmentation for inspection.")


if __name__ == "__main__":
    print("Initializing Image Segmenter for inspection...")
    segmenter = ImageSegmenter(model_name=SEGMENTATION_MODEL_NAME, sky_label_name=DEFAULT_SKY_LABEL)

    if segmenter.segmenter is None:
        print("CRITICAL ERROR: Segmentation model failed to load. Cannot proceed with inspection.")
    else:
        if not os.path.exists(IMAGE_TO_INSPECT):
            print(f"Please update IMAGE_TO_INSPECT in main_inspector.py to a valid image path.")
            print(f"Currently set to: {IMAGE_TO_INSPECT} (Not found)")
        else:
            run_inspection(IMAGE_TO_INSPECT, OUTPUT_VISUALIZATION_PATH, segmenter)

