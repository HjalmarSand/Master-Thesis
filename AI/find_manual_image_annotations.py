import json
import os
import sys

# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to Python's system path
sys.path.append(parent_dir)

import constants


def _get_canonical_polygons(annotations_list):
    """
    Creates a sorted, comparable representation of all polygons for an image.

    Args:
        annotations_list (list): A list of annotation objects for a single image.

    Returns:
        list: A sorted list of tuples, where each tuple is a sorted list of
            the polygon's (x, y) integer points. This is a "canonical"
            representation that can be reliably compared.
    """
    canonical_forms = []
    for ann in annotations_list:
        # COCO segmentation is a list of polygons. We handle the first one.
        flat_coords = ann["segmentation"][0]

        # 1. Group coordinates into (x, y) points and convert to int
        points = []
        for i in range(0, len(flat_coords), 2):
            points.append((int(flat_coords[i]), int(flat_coords[i + 1])))

        # 2. Sort the points to make the polygon's representation
        #    independent of its starting vertex.
        points.sort()

        # 3. Add the canonical form (as a tuple to make it hashable/sortable)
        canonical_forms.append(tuple(points))

    # 4. Sort the list of all canonical polygons to make it independent
    #    of annotation creation order.
    canonical_forms.sort()

    return canonical_forms


def filter_manual_annotations(
    pre_annotation_path, exported_annotation_path, output_path
):
    """
    Compares pre-annotations with exported CVAT annotations to find manual changes.
    """
    print("Loading pre-annotations...")
    with open(pre_annotation_path, "r") as f:
        pre_data = json.load(f)

    print("Loading exported annotations from CVAT...")
    with open(exported_annotation_path, "r") as f:
        exported_data = json.load(f)

    # --- Step 1: Map pre-annotations for easy lookup ---
    pre_annot_map = {}
    pre_images_map = {img["id"]: img for img in pre_data["images"]}

    # Corrected loop iterates through the image map correctly
    for image_id, image_info in pre_images_map.items():
        file_name = image_info["file_name"]

        # Find all annotations associated with this image ID
        annots_for_image = [
            ann for ann in pre_data["annotations"] if ann["image_id"] == image_id
        ]

        # Store the canonical form using the filename as the key
        pre_annot_map[file_name] = _get_canonical_polygons(annots_for_image)

    # --- Step 2: Identify manually changed images ---
    manually_annotated_image_ids = set()

    for image in exported_data["images"]:
        image_id = image["id"]
        file_name = image["file_name"]

        exported_annots_for_image = [
            ann for ann in exported_data["annotations"] if ann["image_id"] == image_id
        ]

        # Case 1: The image was not in pre-annotations, so it's manually annotated.
        if file_name not in pre_annot_map:
            if exported_annots_for_image:  # Check if it actually has annotations
                print(f"Found new manually annotated image: {file_name}")
                manually_annotated_image_ids.add(image_id)
            continue

        # Case 2: Compare polygons with the pre-annotations using the canonical form.
        exported_canonical_polygons = _get_canonical_polygons(exported_annots_for_image)

        pre_annotated_polygons = pre_annot_map.get(file_name, [])

        if exported_canonical_polygons != pre_annotated_polygons:
            print(f"Found modified annotations for image: {file_name}")
            manually_annotated_image_ids.add(image_id)

    # --- Step 3: Build the new COCO JSON file ---
    final_coco = {
        "info": exported_data.get("info", {}),
        "licenses": exported_data.get("licenses", []),
        "categories": exported_data["categories"],
        "images": [],
        "annotations": [],
    }

    # Add only the images and annotations that were manually changed
    for image in exported_data["images"]:
        if image["id"] in manually_annotated_image_ids:
            final_coco["images"].append(image)

    for ann in exported_data["annotations"]:
        if ann["image_id"] in manually_annotated_image_ids:
            final_coco["annotations"].append(ann)

    # --- Step 4: Save the final output ---
    print(f"\nFound {len(final_coco['images'])} manually annotated or modified images.")
    print(f"Saving the filtered annotations to {output_path}")

    with open(output_path, "w") as f:
        json.dump(final_coco, f, indent=4)



def filter_unannotated_images(exported_annotation_path, output_path):
    """
    Filters a COCO JSON file to remove all images that have no annotations.

    Args:
        input_annotation_path (str): Path to the COCO annotation file to be filtered.
        output_path (str): Path to save the new, filtered COCO JSON file.
    """
    print(f"Loading annotations from {input_annotation_path}...")
    with open(input_annotation_path, "r") as f:
        data = json.load(f)

    # Find all image IDs that have annotations
    annotated_image_ids = set()
    for ann in data["annotations"]:
        annotated_image_ids.add(ann["image_id"])

    print(f"Found {len(annotated_image_ids)} images with annotations.")

    final_coco = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [],
        "annotations": [],
    }

    # Add only the images that have a corresponding annotation
    for image in data["images"]:
        if image["id"] in annotated_image_ids:
            final_coco["images"].append(image)

    # All existing annotations should belong to an annotated image,
    # but we filter them as well for correctness.
    for ann in data["annotations"]:
        if ann["image_id"] in annotated_image_ids:
            final_coco["annotations"].append(ann)

    print(
        f"Saving filtered annotations for {len(final_coco['images'])} images to {output_path}"
    )
    with open(output_path, "w") as f:
        json.dump(final_coco, f, indent=4)



# NOTE: Change paths here if for different datasets
if __name__ == "__main__":

    input_annotation_path=constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations/dataset1/annotations/instances_default.json"
    output_path=constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations/dataset1/manual_annotations.json"

    print("")
    process_pre_annotations = input("Enter 1 for filtering of pre-annotated data before removing non-annotated data and 2 if only manually annotated data should be filtered: ")  

    if process_pre_annotations == "1":
        process_pre_annotations = True
    elif process_pre_annotations == "2":
        process_pre_annotations = False

    if process_pre_annotations:
        filter_manual_annotations(
            pre_annotation_path=constants.PROJECT_ROOT
            / "AI/segformer_segmentation_annotations/dataset3/annotations.json",
            exported_annotation_path=input_annotation_path,
            output_path=output_path
        )

    else:
        filter_unannotated_images(
            exported_annotation_path=input_annotation_path,
            output_path=output_path,
        )
