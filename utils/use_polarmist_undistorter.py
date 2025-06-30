import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

import constants
from .polarmist_fisheye import FishCalib

def generate_maps(calib_obj, map_type, cache_path):
    """A helper to generate or load transformation maps from a cache."""
    if os.path.exists(cache_path):
        print(f"Loading cached {map_type} map from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print(f"Cache not found. Generating new {map_type} map...")
    if map_type == 'eqmap':
        maps = calib_obj.gen_eqmaps()
    elif map_type == 'cubemap':
        maps = calib_obj.gen_cubemaps()
    else:
        return None, None
        
    with open(cache_path, "wb") as f:
        pickle.dump(maps, f)
    print(f"Saved new map to: {cache_path}")
    return maps

def process_camera_folder(calib_obj, input_folder, output_folder, remap_function):
    """Processes all images in a given folder."""
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}. Skipping.")
        return
        
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_folder}. Skipping.")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    
    for filename in tqdm(image_files, desc=f"Processing {input_folder.name}"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        distorted_image = cv2.imread(input_path)
        if distorted_image is None:
            continue
            
        undistorted_image = remap_function(distorted_image)
        
        cv2.imwrite(output_path, undistorted_image)

def main():
    """Main function for the fisheye undistortion pipeline stage."""
    os.makedirs(constants.CAM1_UNDISTORTED_FOLDER, exist_ok=True)
    os.makedirs(constants.CAM2_UNDISTORTED_FOLDER, exist_ok=True)
    os.makedirs(constants.CACHE_FOLDER, exist_ok=True)

    print("Initializing fisheye transformer with parameters from constants.py...")
    fish_calib = FishCalib(
        radius=constants.FISHEYE_LENS['radius'],
        aperture=constants.FISHEYE_LENS['aperture_deg'] * np.pi / 180,
        poly_r_denorm2r_ideal=constants.FISHEYE_LENS['poly_coeffs'],
        xc=constants.FISHEYE_LENS['xc'],
        yc=constants.FISHEYE_LENS['yc'],
        sq_size=constants.UNDISTORTION_SETTINGS['output_height']
    )

    mode = constants.UNDISTORTION_SETTINGS['mode']
    if mode == 'equirectangular':
        map_cache_path = constants.CACHE_FOLDER / f"eqmap_{fish_calib.sq_size}.pickle"
        fish_calib.eqmaps = generate_maps(fish_calib, 'eqmap', map_cache_path)
        remap_function = fish_calib.image2eqmap
    elif mode == 'cubemap':
        map_cache_path = constants.CACHE_FOLDER / f"cubemap_{fish_calib.sq_size}.pickle"
        fish_calib.cubemaps = generate_maps(fish_calib, 'cubemap', map_cache_path)
        remap_function = fish_calib.image2cubemap
    else:
        print(f"Error: Unknown undistortion mode '{mode}' in constants.py")
        return 1

    print("\n--- Processing CAM1 ---")
    process_camera_folder(fish_calib, constants.CAM1_DISTORTED_FOLDER, constants.CAM1_UNDISTORTED_FOLDER, remap_function)
    
    print("\n--- Processing CAM2 ---")
    process_camera_folder(fish_calib, constants.CAM2_DISTORTED_FOLDER, constants.CAM2_UNDISTORTED_FOLDER, remap_function)
    
    return 0

if __name__ == '__main__':
    main()