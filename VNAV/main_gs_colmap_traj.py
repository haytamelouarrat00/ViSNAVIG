import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers import DVSController
from scenes import GaussianScene
from cameras import Camera
from control import trajectory_servoing_loop
from controllers.fbvs import FBVSController

def run_gs_colmap_traj(start_idx=0, end_idx=None):
    """
    Setup using GS Scene and COLMAP Poses for a full trajectory.
    """
    print("--- Running GS + COLMAP Trajectory Setup ---")
    gs_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/gskitchen.ply"
    images_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    reconstruction_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/sparse/0"
    
    import pycolmap
    print(f"Loading COLMAP reconstruction from {reconstruction_path}...")
    try:
        reconstruction = pycolmap.Reconstruction(reconstruction_path)
    except Exception as e:
        print(f"Failed to load COLMAP reconstruction: {e}")
        return None, None, None

    posed_images = sorted(reconstruction.images.values(), key=lambda img: img.name)
    
    if end_idx is not None:
        posed_images = posed_images[:end_idx]

    if not posed_images or start_idx >= len(posed_images):
        print(f"No posed images found or start_idx ({start_idx}) >= len ({len(posed_images)}).")
        return None, None, None

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
        return None, None, None
    scene.load(gs_path)
    
    trajectory = []
    print(f"Building trajectory with {len(posed_images) - start_idx - 1} steps...")
    
    for i in range(start_idx + 1, len(posed_images)):
        target_image_info = posed_images[i]
        target_image_path = os.path.join(images_dir, target_image_info.name)
        
        try:
            real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
        except FileNotFoundError:
            print(f"Error: Real target image '{target_image_path}' not found. Stopping trajectory.")
            break

        target_cam = Camera.from_colmap(colmap_cameras_path, camera_id=target_image_info.camera_id)
        try:
            target_cam.set_pose_from_colmap(colmap_images_path, image_name=target_image_info.name)
            target_pose = target_cam.pose
        except Exception as e:
            print(f"Warning: Target pose could not be loaded. {e}")
            target_pose = None
            
        trajectory.append((real_target_img, target_pose))

    return scene, cam, trajectory


def main():
    parser = argparse.ArgumentParser(description="Run GS + COLMAP Trajectory")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index for the trajectory")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending index for the trajectory")
    args = parser.parse_args()

    scene, cam, trajectory = run_gs_colmap_traj(start_idx=args.start_idx, end_idx=args.end_idx)
    
    if scene is None or cam is None or not trajectory:
        print("Failed to initialize setup.")
        return

    # controller = FBVSController(
    #     lambda_gain=1.5,
    #     max_velocity=1.0,
    #     use_moge=True,
    #     ratio=0
    # )
    controller = DVSController(
        lambda_gain=1.5,
    )

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RUNS",
        run_stamp,
    )
    print(f"Run outputs will be written to: {output_dir}")

    trajectory_servoing_loop(
        scene=scene,
        camera=cam,
        trajectory=trajectory,
        controller=controller,
        max_iterations_per_target=300,
        dt=1.0,
        output_dir=output_dir,
        save_frames=True,
        save_trajectory=True,
        run_evo=True,
        frame_format="jpg",
        frame_quality=90,
        error_tolerance=0.0,
        velocity_epsilon=1e-4,
    )

if __name__ == "__main__":
    main()
