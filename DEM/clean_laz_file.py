# Takes a directory of .laz files and cleans them by removing points where the z-value is below a certain threshold.
import os
import argparse
import laspy
from pathlib import Path


def clean_laz_file(input_path, output_path, z_threshold):
    """
    Clean a single LAZ file by removing points below z_threshold.
    
    Args:
        input_path (str): Path to input .laz file
        output_path (str): Path to output .laz file
        z_threshold (float): Minimum z-value to keep
    
    Returns:
        tuple: (original_point_count, filtered_point_count)
    """
    try:
        # Read the LAZ file
        with laspy.open(input_path) as laz_file:
            las = laz_file.read()
        
        original_count = len(las.points)
        
        # Filter points where z >= threshold
        mask = las.z >= z_threshold
        filtered_las = las[mask]
        
        filtered_count = len(filtered_las.points)
        
        # Write the filtered points to output file
        filtered_las.write(output_path)
        
        return original_count, filtered_count
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None, None


def process_directory(input_dir, output_dir, z_threshold, overwrite=False):
    """
    Process all .laz files in a directory.
    
    Args:
        input_dir (str): Input directory containing .laz files
        output_dir (str): Output directory for cleaned files
        z_threshold (float): Minimum z-value to keep
        overwrite (bool): Whether to overwrite existing output files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .laz files
    laz_files = list(input_path.glob("*.laz"))
    
    if not laz_files:
        print(f"No .laz files found in {input_dir}")
        return
    
    print(f"Found {len(laz_files)} .laz files to process")
    print(f"Z-threshold: {z_threshold}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    total_original = 0
    total_filtered = 0
    processed_files = 0
    
    for laz_file in laz_files:
        output_file = output_path / laz_file.name
        
        # Skip if output file exists and overwrite is False
        if output_file.exists() and not overwrite:
            print(f"Skipping {laz_file.name} - output file already exists")
            continue
        
        print(f"Processing: {laz_file.name}")
        
        original_count, filtered_count = clean_laz_file(
            str(laz_file), 
            str(output_file), 
            z_threshold
        )
        
        if original_count is not None:
            removed_count = original_count - filtered_count
            removal_percent = (removed_count / original_count) * 100
            
            print(f"  Original points: {original_count:,}")
            print(f"  Filtered points: {filtered_count:,}")
            print(f"  Removed points: {removed_count:,} ({removal_percent:.1f}%)")
            
            total_original += original_count
            total_filtered += filtered_count
            processed_files += 1
        
        print()
    
    # Summary
    if processed_files > 0:
        total_removed = total_original - total_filtered
        total_removal_percent = (total_removed / total_original) * 100
        
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Files processed: {processed_files}")
        print(f"Total original points: {total_original:,}")
        print(f"Total filtered points: {total_filtered:,}")
        print(f"Total removed points: {total_removed:,} ({total_removal_percent:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean LAZ files by removing points below a z-threshold (i.e points that are lower than water)."
    )
    
    parser.add_argument(
        "--input_dir", 
        required=True,
        help="Directory containing .laz files to process"
    )
    
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Directory to save cleaned .laz files"
    )
    
    parser.add_argument(
        "--z_threshold", 
        type=float, 
        required=True,
        help="Minimum z-value to keep (points below this will be removed)"
    )
    
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing output files"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Process the directory
    process_directory(
        args.input_dir, 
        args.output_dir, 
        args.z_threshold, 
        args.overwrite
    )


if __name__ == "__main__":
    main()