import os
import sys
import time
import numpy as np
from PIL import Image
import glob

# Ensure the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenes import MeshScene
from cameras import Camera
from control import visual_servoing_loop
from controllers.fbvs import FBVSController

def run_mesh_scannet_step(start_idx=0, next_idx=1):
    """
    Setup using Mesh Scene and ScanNet Poses for a single step (one index to the next).
    """
    print(f"--- Running Mesh + ScanNet Setup (Step: {start_idx} -> {next_idx}) ---")
    mesh_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/akitchen.ply"
    info_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/info.txt"
    data_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    
    color_files = sorted(glob.glob(os.path.join(data_dir, "*.color.jpg")))
    if not color_files or start_idx >= len(color_files) or next_idx >= len(color_files):
        print("Invalid indices or no images found in", data_dir)
        return None, None, None, None
        
    start_image_path = color_files[start_idx]
    start_pose_path = start_image_path.replace(".color.jpg", ".pose.txt")
    
    print("Loading Intrinsics...")
    cam = Camera.from_dataset_info(info_path, sensor_type='color')
    
    print("Loading Start Pose...")
    cam.set_pose_from_scannet(start_pose_path)
    
    print("Initializing Mesh Scene...")
    scene = MeshScene(width=cam.width, height=cam.height)
    if not os.path.exists(mesh_path):
        print(f"Please update paths. '{mesh_path}' not found.")
        return None, None, None, None
    scene.load(mesh_path)
    
    target_image_path = color_files[next_idx]
    target_pose_path = target_image_path.replace(".color.jpg", ".pose.txt")
    
    try:
        real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
    except FileNotFoundError:
        print(f"Error: Real target image '{target_image_path}' not found.")
        return None, None, None, None

    target_cam = Camera.from_dataset_info(info_path, sensor_type='color')
    try:
        target_cam.set_pose_from_scannet(target_pose_path)
        target_pose = target_cam.pose
    except FileNotFoundError:
        print(f"Warning: Target pose '{target_pose_path}' not found. Distance metrics will not be available.")
        target_pose = None
        
    return scene, cam, real_target_img, target_pose


def main():
    # You can change the indices here to test different steps
    scene, cam, target_image, target_pose = run_mesh_scannet_step(start_idx=0, next_idx=1)
    
    if scene is None or cam is None or target_image is None:
        print("Failed to initialize setup.")
        return

    controller = FBVSController(
        lambda_gain=1.5,
        max_velocity=1.0,
        use_moge=True,
        ratio=0
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
        dt=0.1,
        output_dir=output_dir,
        save_frames=False,
        save_trajectory=True,
        run_evo=True,
        frame_format="jpg",
        frame_quality=90,
    )

if __name__ == "__main__":
    main()
