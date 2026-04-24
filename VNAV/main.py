import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import glob

# Ensure the parent directory is in the Python path to allow `import *`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenes import GaussianScene, MeshScene
from cameras import Camera
from control import visual_servoing_loop, trajectory_servoing_loop
from controllers.fbvs import FBVSController

def run_3dgs_colmap():
    """
    Setup using 3D Gaussian Splatting and COLMAP Poses.
    """
    print("--- Running 3DGS + COLMAP Setup ---")
    gs_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/gskitchen.ply"
    images_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    reconstruction_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/sparse/0"
    
    import pycolmap
    print(f"Loading COLMAP reconstruction from {reconstruction_path}...")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)

    # Sort images by their filename (name) to guarantee temporal and spatial sequentiality
    posed_images = sorted(reconstruction.images.values(), key=lambda img: img.name)

    # 1. Pick Start Pose
    start_index = 0
    start_image = posed_images[start_index]
    colmap_camera_id = start_image.camera_id

    colmap_cameras_path = os.path.join(reconstruction_path, "cameras.bin")
    colmap_images_path = os.path.join(reconstruction_path, "images.bin")

    print(f"Initial Pose: '{start_image.name}' (Camera ID: {colmap_camera_id})")
    cam = Camera.from_colmap(colmap_cameras_path, camera_id=colmap_camera_id)
    cam.set_pose_from_colmap(colmap_images_path, image_name=start_image.name)

    # 2. Initialize Scene
    print("Initializing Gaussian Scene...")
    scene = GaussianScene(width=cam.width, height=cam.height)
    if not os.path.exists(gs_path):
        print(f"Please update paths. '{gs_path}' not found.")
        return None, None, None
    scene.load(gs_path)
    
    # 3. Build trajectory
    trajectory = []
    print(f"Building trajectory with {len(posed_images) - start_index - 1} steps...")
    
    for i in range(start_index + 1, len(posed_images)):
        target_image_info = posed_images[i]
        target_image_path = os.path.join(images_dir, target_image_info.name)
        
        try:
            real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
        except FileNotFoundError:
            print(f"Error: Real target image '{target_image_path}' not found. Stopping trajectory.")
            break

        target_cam = Camera.from_colmap(colmap_cameras_path, camera_id=target_image_info.camera_id)
        target_cam.set_pose_from_colmap(colmap_images_path, image_name=target_image_info.name)
        target_pose = target_cam.pose
        
        trajectory.append((real_target_img, target_pose))

    return scene, cam, trajectory


def run_mesh_scannet():
    """
    Setup using Mesh Scene and ScanNet Poses.
    """
    print("--- Running Mesh + ScanNet Setup ---")
    mesh_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/akitchen.ply"
    info_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/info.txt"
    data_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    
    color_files = sorted(glob.glob(os.path.join(data_dir, "*.color.jpg")))
    if not color_files:
        print("No color images found in", data_dir)
        return None, None, None
        
    start_image_path = color_files[0]
    start_pose_path = start_image_path.replace(".color.jpg", ".pose.txt")
    
    # 1. Initialize Camera with Intrinsics from info.txt
    print("Loading Intrinsics...")
    cam = Camera.from_dataset_info(info_path, sensor_type='color')
    
    # 2. Set Start Pose
    print("Loading Start Pose...")
    cam.set_pose_from_scannet(start_pose_path)
    
    # 3. Initialize Scene
    print("Initializing Mesh Scene...")
    scene = MeshScene(width=cam.width, height=cam.height)
    if not os.path.exists(mesh_path):
        print(f"Please update paths. '{mesh_path}' not found.")
        return None, None, None
    scene.load(mesh_path)
    
    # 4. Build trajectory
    trajectory = []
    print(f"Building trajectory with {len(color_files) - 1} steps...")
    
    for i in range(1, len(color_files)):
        target_image_path = color_files[i]
        target_pose_path = target_image_path.replace(".color.jpg", ".pose.txt")
        
        try:
            real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
        except FileNotFoundError:
            print(f"Error: Real target image '{target_image_path}' not found.")
            break

        target_cam = Camera.from_dataset_info(info_path, sensor_type='color')
        try:
            target_cam.set_pose_from_scannet(target_pose_path)
            target_pose = target_cam.pose
        except FileNotFoundError:
            print(f"Warning: Target pose '{target_pose_path}' not found. Distance metrics will not be available.")
            target_pose = None
            
        trajectory.append((real_target_img, target_pose))
        
    return scene, cam, trajectory


def _parse_args():
    parser = argparse.ArgumentParser(description="VISNAV visual servoing driver.")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=False,
        help="Save a [render|target] image every iteration (OFF by default; adds disk I/O).",
    )
    parser.add_argument(
        "--frame-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Frame image format when --save-frames is on. jpg is fastest (default).",
    )
    parser.add_argument(
        "--frame-quality",
        type=int,
        default=90,
        help="JPEG quality 1-100 (ignored for png). Default 90.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # =========================================================================
    # CHOOSE YOUR SETUP
    # Uncomment the one you want to run.
    # =========================================================================

    # [Option A] Run with 3DGS and COLMAP poses
    # scene, cam, trajectory = run_3dgs_colmap()

    # [Option B] Run with Mesh and ScanNet poses
    scene, cam, trajectory = run_mesh_scannet()

    if scene is None or cam is None or not trajectory:
        print("Failed to initialize setup.")
        return

    # =========================================================================
    # INITIALIZE THE CONTROLLER
    # =========================================================================
    # We instantiate the Feature-Based Visual Servoing Controller.
    # It uses XFeat for matching and MoGe-2 to estimate the depth of the virtual view.
    controller = FBVSController(
        lambda_gain=1.5,     # Adjust for convergence speed
        max_velocity=1.0,    # Max camera speed limit
        use_moge=True,       # True: Use MoGe-2, False: Use Ground Truth Z-Buffer
        ratio=0              # 0 = detect once and reproject, 1 = redetect every frame, N = redetect every N frames
    )

    # =========================================================================
    # START THE VISUAL SERVOING LOOP
    # =========================================================================
    # Each run is saved under RUNS/<timestamp>/ so trajectories (and optionally
    # frames) are preserved for later inspection and evo-based evaluation.
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RUNS",
        run_stamp,
    )
    print(f"Run outputs will be written to: {output_dir}")
    print(f"save_frames={args.save_frames} (format={args.frame_format}, quality={args.frame_quality})")

    trajectory_servoing_loop(
        scene=scene,
        camera=cam,
        trajectory=trajectory,
        controller=controller,
        max_iterations_per_target=300,
        dt=0.1,
        output_dir=output_dir,
        save_frames=args.save_frames,
        save_trajectory=True,
        run_evo=True,
        frame_format=args.frame_format,
        frame_quality=args.frame_quality,
    )

if __name__ == "__main__":
    main()
