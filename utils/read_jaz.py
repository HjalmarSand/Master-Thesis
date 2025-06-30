import numpy as np
import laspy
import collections

# Define the file path
# Using a variable makes it easier to change later
file_path = 'lantmateri_data/y657_69_7500_17.laz'

# Use a dictionary or collections.Counter for more flexible counting
ground_return_counts = collections.Counter()

try:
    with laspy.open(file_path) as f:
        print(f'File: {file_path}')
        print(f'Points from Header: {f.header.point_count}')
        print(f'Point format: {f.header.point_format.id}')
        print(f'CRS: {f.header.crs}') # Good to check the coordinate system
        
        # Iterate over the points in chunks to save memory
        for points in f.chunk_iterator(1_000_000): # Process 1 million points at a time
            # Create a boolean mask for ground points in the current chunk
            is_ground = points.classification == 2
            
            # Get the return numbers for only the ground points
            ground_returns = points.return_number[is_ground]
            
            # Update our counter with the counts from this chunk
            ground_return_counts.update(ground_returns)

    # --- Analysis after reading ---
    total_ground_points = sum(ground_return_counts.values())
    print(f'\nTotal ground points found: {total_ground_points}')
    print('Ground Point Return Number distribution:')
    
    # Sort by return number (bin) for clean printing
    for r, c in sorted(ground_return_counts.items()):
        print(f'    Return Number {r}: {c} points')

except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'")
    print("Please make sure the script is run from the correct directory or use an absolute path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")