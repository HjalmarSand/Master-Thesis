import subprocess
import shutil
from pathlib import Path
import config

COLMAP_EXE_PATH = shutil.which("colmap")

if COLMAP_EXE_PATH:
    print(f"Using Colmap from system PATH: {COLMAP_EXE_PATH}")
else:
    print(f"'colmap' not found in system PATH. Assuming 'colmap' command will resolve.")
    print(f"If Colmap calls fail, ensure 'colmap' is in your system PATH or manually set COLMAP_EXE_PATH.")
    COLMAP_EXE_PATH = "colmap" # Default to "colmap" command

# --- Camera Hardware Specifications ---
SENSOR_FOCAL_LENGTH_MM = 0.87
SENSOR_PIXEL_SIZE_MM = 0.00112

# --- Native Sensor Dimensions (for scaling calculation) ---
NATIVE_SENSOR_WIDTH_PX = 3280
# NATIVE_SENSOR_HEIGHT_PX = 2464

# --- Main Colmap Processing Function ---
def estimate_camera_parameters(camera_name: str,
                                image_dir: Path,
                                base_workspace_dir: Path, # This is config.COLMAP_DIR
                                image_width: int,         # From config.SPLIT_IMG_WIDTH
                                image_height: int):       # From config.SPLIT_IMG_HEIGHT
    
    print(f"\n--- Estimating parameters for {camera_name} ---")
    print(f"--- Images from: {image_dir} ---")
    print(f"--- Using configured image dimensions: {image_width}x{image_height} ---")

    workspace_dir = base_workspace_dir / f"workspace_{camera_name}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / f"database_{camera_name}.db"
    sparse_model_dir = workspace_dir / f"sparse_{camera_name}"

    fx_pixels_native_res = SENSOR_FOCAL_LENGTH_MM / SENSOR_PIXEL_SIZE_MM
    width_scale_factor = image_width / NATIVE_SENSOR_WIDTH_PX
    initial_fx_guess = fx_pixels_native_res * width_scale_factor
    initial_fy_guess = initial_fx_guess 
    initial_cx_guess = image_width / 2.0
    initial_cy_guess = image_height / 2.0
    initial_k1, initial_k2, initial_k3, initial_k4 = 0.0, 0.0, 0.0, 0.0

    initial_params_str = (
        f"{initial_fx_guess:.4f},{initial_fy_guess:.4f},"
        f"{initial_cx_guess:.2f},{initial_cy_guess:.2f},"
        f"{initial_k1},{initial_k2},{initial_k3},{initial_k4}"
    )
    print(f"Initial guess for OPENCV_FISHEYE: {initial_params_str}")
    
    print(f"\n[ {camera_name} ] Running Feature Extraction...")
    cmd_feature_extraction = [
        COLMAP_EXE_PATH, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "OPENCV_FISHEYE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", initial_params_str,
        "--SiftExtraction.use_gpu", "0", # 1 for GPU, not supported on my machine
    ]
    try:
        subprocess.run(cmd_feature_extraction, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Feature extraction successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error during feature extraction: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: Colmap executable '{COLMAP_EXE_PATH}' not found. Is it in PATH?")
        return None

    print(f"\n[ {camera_name} ] Running Feature Matching...")
    cmd_feature_matching = [
        COLMAP_EXE_PATH, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1",
    ]
    try:
        subprocess.run(cmd_feature_matching, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Feature matching successful.")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error during feature matching: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None

    print(f"\n[ {camera_name} ] Running Incremental Reconstruction (Mapping)...")
    sparse_model_dir.mkdir(parents=True, exist_ok=True)
    cmd_mapper = [
        COLMAP_EXE_PATH, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_model_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ]
    try:
        subprocess.run(cmd_mapper, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Incremental reconstruction successful. Model in {sparse_model_dir / '0'}")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error during mapping: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None

    print(f"\n[ {camera_name} ] Extracting Camera Parameters...")
    reconstruction_path_bin_folder = sparse_model_dir / "0"
    
    if not (reconstruction_path_bin_folder / "cameras.bin").exists():
        print(f"[ {camera_name} ] cameras.bin not found in {reconstruction_path_bin_folder}. Mapping might have failed.")

    cmd_model_converter = [
        COLMAP_EXE_PATH, "model_converter",
        "--input_path", str(reconstruction_path_bin_folder),
        "--output_path", str(reconstruction_path_bin_folder),
        "--output_type", "TXT"
    ]
    try:
        subprocess.run(cmd_model_converter, check=True, capture_output=True, text=True)
        print(f"[ {camera_name} ] Converted model in {reconstruction_path_bin_folder} to TXT format.")
    except subprocess.CalledProcessError as e:
        print(f"[ {camera_name} ] Error converting model to TXT: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return None

    cameras_txt_path = reconstruction_path_bin_folder / "cameras.txt"
    extracted_params = {}
    if cameras_txt_path.exists():
        with open(cameras_txt_path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) >= 12 and parts[1] == "OPENCV_FISHEYE":
                    extracted_params = {
                        "camera_id_colmap": int(parts[0]), "model": parts[1], 
                        "width": int(parts[2]), "height": int(parts[3]),
                        "fx": float(parts[4]), "fy": float(parts[5]), 
                        "cx": float(parts[6]), "cy": float(parts[7]),
                        "k1": float(parts[8]), "k2": float(parts[9]), 
                        "k3": float(parts[10]), "k4": float(parts[11])
                    }
                    print(f"\nEstimated Parameters for {camera_name}:")
                    for key, val in extracted_params.items():
                        if isinstance(val, float): print(f"  {key}: {val:.4f}")
                        else: print(f"  {key}: {val}")
                    break 
    else:
        print(f"[ {camera_name} ] cameras.txt not found at {cameras_txt_path} after conversion attempt.")
        return None
    
    return extracted_params

# --- Main Script Logic ---
if __name__ == "__main__":
    config.COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    all_estimated_params = {}

    # Process Camera 1
    print(f"\n--- Starting processing for Camera 1 ---")
    cam1_image_folder = config.COLMAP_DIR / "images_cam1"
    if cam1_image_folder.is_dir(): 
        params_cam1 = estimate_camera_parameters(
            "cam1", 
            cam1_image_folder, 
            config.COLMAP_DIR,
            config.SPLIT_IMG_WIDTH, 
            config.SPLIT_IMG_HEIGHT
        )
        if params_cam1:
            all_estimated_params["cam1"] = params_cam1
    else:
        print(f"Skipping Cam1: Image folder '{cam1_image_folder}' not found or not a directory.")

    # Process Camera 2
    print(f"\n--- Starting processing for Camera 2 ---")
    cam2_image_folder = config.COLMAP_DIR / "images_cam2"
    if cam2_image_folder.is_dir():
        params_cam2 = estimate_camera_parameters(
            "cam2", 
            cam2_image_folder, 
            config.COLMAP_DIR,
            config.SPLIT_IMG_WIDTH, 
            config.SPLIT_IMG_HEIGHT
        )
        if params_cam2:
            all_estimated_params["cam2"] = params_cam2
    else:
        print(f"Skipping Cam2: Image folder '{cam2_image_folder}' not found or not a directory.")



    # --- Final Summary ---
    print("\n\n" + "="*30 + " FINAL SUMMARY " + "="*30)

    def print_camera_summary(cam_name: str, params_dict_or_none, img_width_cfg: int, img_height_cfg: int, base_dir_cfg: Path):
        print(f"\n--- {cam_name.upper()} (Configured Dims: {img_width_cfg}x{img_height_cfg}) ---")
        if params_dict_or_none:
            print("  Initial Guess (Spec-based for OPENCV_FISHEYE):")
            fx_native = SENSOR_FOCAL_LENGTH_MM / SENSOR_PIXEL_SIZE_MM
            w_scale = img_width_cfg / NATIVE_SENSOR_WIDTH_PX
            fx_s, fy_s = fx_native * w_scale, fx_native * w_scale
            cx_s, cy_s = img_width_cfg / 2.0, img_height_cfg / 2.0
            print(f"    fx: {fx_s:.2f}, fy: {fy_s:.2f}, cx: {cx_s:.2f}, cy: {cy_s:.2f}, k1-k4: 0,0,0,0")
            
            print("  Final Estimated Parameters (from Colmap):")
            for k, v in params_dict_or_none.items():
                if isinstance(v, float): print(f"    {k}: {v:.4f}")
                else: print(f"    {k}: {v}")
            print(f"  Workspace: {base_dir_cfg / f'workspace_{cam_name}'}")
        else:
            print("  Parameters not estimated or estimation failed.")

    print_camera_summary("cam1", all_estimated_params.get("cam1"), config.SPLIT_IMG_WIDTH, config.SPLIT_IMG_HEIGHT, config.COLMAP_DIR)
    print_camera_summary("cam2", all_estimated_params.get("cam2"), config.SPLIT_IMG_WIDTH, config.SPLIT_IMG_HEIGHT, config.COLMAP_DIR)

    if not all_estimated_params or (not all_estimated_params.get("cam1") and not all_estimated_params.get("cam2")):
        print("\nNo parameters were successfully estimated for any camera, or all failed.")
    else:
        print("\nParameter estimation complete for at least one camera.")
        print("Check workspace folders for Colmap models (e.g., sparse/0/cameras.bin).")

    print("\n--- Script Finished ---")