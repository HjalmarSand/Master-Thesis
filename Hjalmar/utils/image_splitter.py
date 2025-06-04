import os
from PIL import Image

def split_images_in_folder(input_folder, output_folder):
    """
    Splits all JPG images in the input_folder vertically in the middle.
    The upper half is saved as "_back.jpg" and the lower half as "_front.jpg"
    in the output_folder.

    Args:
        input_folder (str): Path to the folder containing input JPG images.
        output_folder (str): Path to the folder where split images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    processed_count = 0
    error_count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            base_name, ext = os.path.splitext(filename)

            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    mid_y = height // 2  # Integer division to find the middle y-coordinate

                    # Upper part of the original image
                    box_upper = (0, 0, width, mid_y)
                    img_upper = img.crop(box_upper)

                    # Lower part of the original image
                    box_lower = (0, mid_y, width, height)
                    img_lower = img.crop(box_lower)

                    # Lower part will be "_front"
                    output_path_front = os.path.join(output_folder, f"{base_name}_front{ext}")
                    # Upper part will be "_back"
                    output_path_back = os.path.join(output_folder, f"{base_name}_back{ext}")

                    img_lower.save(output_path_front) 
                    img_upper.save(output_path_back)  

                    print(f"Processed: {filename} -> {os.path.basename(output_path_front)}, {os.path.basename(output_path_back)}")
                    processed_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                error_count += 1
        else:
            if os.path.isfile(os.path.join(input_folder, filename)): 
                print(f"Skipping non-JPG file: {filename}")

    print(f"\n--- Summary ---")
    print(f"Total images processed: {processed_count}")
    if error_count > 0:
        print(f"Images with errors: {error_count}")
    print(f"Split images saved in: {output_folder}")

if __name__ == "__main__":
    INPUT_IMAGE_FOLDER = "dataset1/images"
    OUTPUT_IMAGE_FOLDER = "dataset1/split_images"

    if not os.path.isdir(INPUT_IMAGE_FOLDER):
        print(f"Error: Input folder '{INPUT_IMAGE_FOLDER}' does not exist.")
        print("Please create it and place your JPG images inside, or correct the path.")
    else:
        split_images_in_folder(INPUT_IMAGE_FOLDER, OUTPUT_IMAGE_FOLDER)