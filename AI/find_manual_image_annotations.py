import json
import sys
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
# Add the parent directory to Python's system path
sys.path.append(str(parent_dir))

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
        if not ann["segmentation"]:
            continue
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

    # 4. Sort the list of all canonical polygons to make it independent of annotation creation order.
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

    # Step 1: Map pre-annotations for easy lookup
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

    # Step 2: Identify manually changed images
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

    # Step 3: Build the new COCO JSON file
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

    print(f"\nFound {len(final_coco['images'])} manually annotated or modified images.")
    print(f"Saving the filtered annotations to {output_path}")

    with open(output_path, "w") as f:
        json.dump(final_coco, f, indent=4)


def filter_unannotated_images(exported_annotation_path, output_path):
    """
    Filters a COCO JSON file to remove all images that have no annotations.

    Args:
        exported_annotation_path (str): Path to the COCO annotation file to be filtered.
        output_path (str): Path to save the new, filtered COCO JSON file.
    """
    print(f"Loading annotations from {exported_annotation_path}...")
    with open(exported_annotation_path, "r") as f:
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


def _select_from_list(items, prompt_message):
    """Helper function to prompt user to select an item from a list."""
    if not items:
        return None

    print(f"\n{prompt_message}")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(items):
                return items[choice - 1]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """Main function to drive the interactive annotation filtering process."""
    base_datasets_dir = constants.PROJECT_ROOT / "AI/segformer_segmentation_annotations"

    if not base_datasets_dir.is_dir():
        print(f"Error: Base directory not found at {base_datasets_dir}")
        sys.exit(1)

    # 1. Find and select the dataset directory
    dataset_dirs = sorted([d.name for d in base_datasets_dir.iterdir() if d.is_dir()])
    if not dataset_dirs:
        print(f"Error: No dataset directories found in {base_datasets_dir}")
        sys.exit(1)

    selected_dir_name = _select_from_list(
        dataset_dirs, "Please select a dataset folder to process:"
    )
    if not selected_dir_name:
        sys.exit(1)  # Should not happen if there are dirs, but good practice

    selected_dataset_path = base_datasets_dir / selected_dir_name

    # 2. Find and select the annotation file to process
    # Exclude the default pre-annotation file and any previously generated manual files
    potential_files = sorted(
        [
            f.name
            for f in selected_dataset_path.glob("*.json")
            if f.name != "annotations.json" and not f.name.startswith("manual_")
        ]
    )

    if not potential_files:
        print(
            f"Error: No suitable exported annotation .json files found in {selected_dataset_path}"
        )
        print(
            "Looking for files that are not named 'annotations.json' or start with 'manual_'."
        )
        sys.exit(1)

    selected_file_name = _select_from_list(
        potential_files, "Please select the exported annotation file to process:"
    )
    if not selected_file_name:
        sys.exit(1)

    # 3. Construct all necessary paths
    exported_annotation_path = selected_dataset_path / selected_file_name
    output_path = selected_dataset_path / f"manual_{selected_file_name}"
    pre_annotation_path = selected_dataset_path / "annotations.json"

    # 4. Select the processing mode
    print("")
    actions = [
        "Filter out pre-annotations (finds manual changes)",
        "Filter out images with no annotations (cleans a single file)",
    ]
    selected_action = _select_from_list(
        actions, "What action would you like to perform?"
    )

    if selected_action == actions[0]:
        # This is "Filter out pre-annotations"
        if not pre_annotation_path.is_file():
            print(
                f"\nError: Pre-annotation file 'annotations.json' not found in {selected_dataset_path}."
            )
            print("Cannot compare to find manual changes.")
            sys.exit(1)

        filter_manual_annotations(
            pre_annotation_path=pre_annotation_path,
            exported_annotation_path=exported_annotation_path,
            output_path=output_path,
        )
    elif selected_action == actions[1]:
        # This is "Filter out unannotated images"
        filter_unannotated_images(
            exported_annotation_path=exported_annotation_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
