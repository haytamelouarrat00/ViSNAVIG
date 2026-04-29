import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenes import GaussianScene
from cameras import Camera
from control import visual_servoing_loop
from controllers.dvs import DVSController

def run_gs_colmap_step(start_idx=0, next_idx=1):
    """
    Setup using GS Scene and COLMAP Poses for a single step (one index to the next).
    """
    print(f"--- Running GS + COLMAP Setup (Step: {start_idx} -> {next_idx}) ---")
    gs_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/gskitchen.ply"
    images_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    reconstruction_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/sparse/0"
    
    import pycolmap
    print(f"Loading COLMAP reconstruction from {reconstruction_path}...")
    try:
        reconstruction = pycolmap.Reconstruction(reconstruction_path)
    except Exception as e:
        print(f"Failed to load COLMAP reconstruction: {e}")
        return None, None, None, None

    posed_images = sorted(reconstruction.images.values(), key=lambda img: img.name)
    
    if not posed_images or start_idx >= len(posed_images) or next_idx >= len(posed_images):
        print("Invalid indices or no images found in COLMAP reconstruction.")
        return None, None, None, None

    colmap_cameras_path = os.path.join(reconstruction_path, "cameras.bin")
    colmap_images_path = os.path.join(reconstruction_path, "images.bin")
    
    start_image_info = posed_images[start_idx]
    colmap_camera_id = start_image_info.camera_id

    print(f"Initial Pose: '{start_image_info.name}' (Camera ID: {colmap_camera_id})")
    cam = Camera.from_colmap(colmap_cameras_path, camera_id=colmap_camera_id)
    cam.set_pose_from_colmap(colmap_images_path, image_name=start_image_info.name)

    print("Initializing Gaussian Scene...")
    scene = GaussianScene(width=cam.width, height=cam.height)
    if not os.path.exists(gs_path):
        print(f"Please update paths. '{gs_path}' not found.")
        return None, None, None, None
    scene.load(gs_path)
    
    target_image_info = posed_images[next_idx]
    target_image_path = os.path.join(images_dir, target_image_info.name)
    
    try:
        real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
    except FileNotFoundError:
        print(f"Error: Real target image '{target_image_path}' not found.")
        return None, None, None, None

    target_cam = Camera.from_colmap(colmap_cameras_path, camera_id=target_image_info.camera_id)
    try:
        target_cam.set_pose_from_colmap(colmap_images_path, image_name=target_image_info.name)
        target_pose = target_cam.pose
    except Exception as e:
        print(f"Warning: Target pose could not be loaded. {e}")
        target_pose = None
        
    return scene, cam, real_target_img, target_pose


def main():
    scene, cam, target_image, target_pose = run_gs_colmap_step(start_idx=50, next_idx=51)
    
    if scene is None or cam is None or target_image is None:
        print("Failed to initialize setup.")
        return

    controller = DVSController(
        lambda_gain=1.5
    )

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RUNS",
        run_stamp,
    )
    print(f"Run outputs will be written to: {output_dir}")

    visual_servoing_loop(
        scene=scene,
        camera=cam,
        target_image=target_image,
        target_pose=target_pose,
        controller=controller,
        max_iterations=300,
        dt=1.0,
        error_tolerance=0.0,
        velocity_epsilon=1e-4,
        output_dir=output_dir,
        save_frames=True,
        save_trajectory=True,
        run_evo=False,
        frame_format="jpg",
        frame_quality=90,
    )

if __name__ == "__main__":
    main()
