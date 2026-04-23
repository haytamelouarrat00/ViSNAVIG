import os
import sys
import numpy as np
from PIL import Image

# Ensure the parent directory is in the Python path to allow `import *`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenes import GaussianScene, MeshScene
from cameras import Camera
from control import visual_servoing_loop
from controllers.fbvs import FBVSController

def run_3dgs_colmap():
    """
    Setup using 3D Gaussian Splatting and COLMAP Poses.
    We pick a 'start' pose from the reconstruction to initialize the virtual camera,
    and a 'target' real image from another index to servo towards.
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

    # 2. Pick Target Image and Target Pose
    target_index = start_index + 10  # Now guaranteed to be the next spatial frame!
    target_image_info = posed_images[target_index]
    target_image_path = os.path.join(images_dir, target_image_info.name)
    print(f"Target Image: '{target_image_info.name}'")
    
    try:
        real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
    except FileNotFoundError:
        print(f"Error: Real target image '{target_image_path}' not found.")
        return None, None, None, None

    # Load target pose
    target_cam = Camera.from_colmap(colmap_cameras_path, camera_id=target_image_info.camera_id)
    target_cam.set_pose_from_colmap(colmap_images_path, image_name=target_image_info.name)
    target_pose = target_cam.pose

    # 3. Initialize Scene
    print("Initializing Gaussian Scene...")
    scene = GaussianScene(width=cam.width, height=cam.height)
    if not os.path.exists(gs_path):
        print(f"Please update paths. '{gs_path}' not found.")
        return None, None, None, None
    scene.load(gs_path)
    
    return scene, cam, real_target_img, target_pose


def run_mesh_scannet():
    """
    Setup using Mesh Scene and ScanNet Poses.
    We pick a 'start' pose from a .pose.txt file and a 'target' image from another frame.
    """
    print("--- Running Mesh + ScanNet Setup ---")
    mesh_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/akitchen.ply"
    info_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/info.txt"
    start_pose_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data/frame-000631.pose.txt"
    target_image_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data/frame-000640.color.jpg"
    target_pose_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data/frame-000640.pose.txt"
    
    # 1. Initialize Camera with Intrinsics from info.txt
    print("Loading Intrinsics...")
    cam = Camera.from_dataset_info(info_path, sensor_type='color')
    
    # 2. Set Start Pose
    print("Loading Start Pose...")
    cam.set_pose_from_scannet(start_pose_path)
    
    # 3. Load Target Image
    print(f"Target Image: '{target_image_path}'")
    try:
        real_target_img = np.array(Image.open(target_image_path).convert("RGB"))
    except FileNotFoundError:
        print(f"Error: Real target image '{target_image_path}' not found.")
        return None, None, None, None

    # Load target pose
    target_cam = Camera.from_dataset_info(info_path, sensor_type='color')
    try:
        target_cam.set_pose_from_scannet(target_pose_path)
        target_pose = target_cam.pose
    except FileNotFoundError:
        print(f"Warning: Target pose '{target_pose_path}' not found. Distance metrics will not be available.")
        target_pose = None

    # 4. Initialize Scene
    print("Initializing Mesh Scene...")
    scene = MeshScene(width=cam.width, height=cam.height)
    if not os.path.exists(mesh_path):
        print(f"Please update paths. '{mesh_path}' not found.")
        return None, None, None, None
    scene.load(mesh_path)
    
    return scene, cam, real_target_img, target_pose


def main():
    # =========================================================================
    # CHOOSE YOUR SETUP
    # Uncomment the one you want to run.
    # =========================================================================
    
    # [Option A] Run with 3DGS and COLMAP poses
    # scene, cam, real_target_img, target_pose = run_3dgs_colmap()
    
    # [Option B] Run with Mesh and ScanNet poses
    scene, cam, real_target_img, target_pose = run_mesh_scannet()
    
    if scene is None or cam is None or real_target_img is None:
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
    # The loop acts as a simple executor. It passes the views to the controller,
    # receives the velocity, and moves the camera.
    visual_servoing_loop(
        scene=scene,
        camera=cam,
        target_image=real_target_img,
        controller=controller,
        target_pose=target_pose,
        max_iterations=300,
        dt=0.1
    )

if __name__ == "__main__":
    main()
