import cv2
import numpy as np
import os
import time

import constants

from utils.polarmist_fisheye import FishCalib

def test_both_undistortion_types():
    """
    A simple test to undistort a single fisheye image into BOTH an
    equirectangular image and a cubemap.
    """
    print("--- Starting Combined Undistortion Test (Equirectangular & Cubemap) ---")

    output_equirect_path = "test_output_equirectangular.jpg"
    output_cubemap_path = "test_output_cubemap.jpg"

    print("Creating FishCalib instance...")
    poly_coeffs = np.array([0.016248739491802746, 0.018858310252328498, 0.01971739099626771, 0.007914765999675276, 0.934584334780868, 0.0])
    calib_instance = FishCalib(
        radius=662,
        aperture=200 * np.pi / 180,
        poly_r_denorm2r_ideal=poly_coeffs,
        sq_size=512
    )

    print(f"Loading test image: {os.path.join(constants.CAM1_DISTORTED_FOLDER, "distorted_cam1_0801.jpg")}")
    distorted_image = cv2.imread(os.path.join(constants.CAM1_DISTORTED_FOLDER, "distorted_cam1_0801.jpg"))

    if distorted_image is None:
        print(f"!!! ERROR: Could not load the image at '{os.path.join(constants.CAM1_DISTORTED_FOLDER, "cam1_0801.jpg")}'. Please check the path.")
        return
    
    print("\n--- Processing Equirectangular ---")
    
    # Generate the equirectangular mapping.
    print("Generating the equirectangular transformation map...")
    start_time = time.time()
    calib_instance.eqmaps = calib_instance.gen_eqmaps()
    print(f"Equirectangular map generation took {time.time() - start_time:.2f} seconds.")

    # Perform the undistortion using the generated map
    print("Remapping the image...")
    equirect_image = calib_instance.image2eqmap(distorted_image)

    # Save the final output image
    print(f"Saving equirectangular image to: {output_equirect_path}")
    cv2.imwrite(output_equirect_path, equirect_image)

    print("\n--- Processing Cubemap ---")
    
    # Generate the cubemap mapping.
    print("Generating the cubemap transformation map...")
    start_time = time.time()
    calib_instance.cubemaps = calib_instance.gen_cubemaps()
    print(f"Cubemap map generation took {time.time() - start_time:.2f} seconds.")

    # Perform the undistortion using the generated map
    print("Remapping the image...")
    cubemap_image = calib_instance.image2cubemap(distorted_image)

    # Save the final output image
    print(f"Saving cubemap image to: {output_cubemap_path}")
    cv2.imwrite(output_cubemap_path, cubemap_image)


    print("\n--- Test Complete ---")
    print(f"Check the output files:")
    print(f"  - {output_equirect_path}")
    print(f"  - {output_cubemap_path}")


if __name__ == "__main__":
    test_both_undistortion_types()