import argparse
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
from control import trajectory_servoing_loop
from controllers.fbvs import FBVSController

def run_mesh_scannet_traj(start_idx=0, end_idx=None):
    """
    Setup using Mesh Scene and ScanNet Poses for a full trajectory.
    """
    print("--- Running Mesh + ScanNet Trajectory Setup ---")
    mesh_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/akitchen.ply"
    info_path = "/home/haytam-elourrat/VISNAV/DATA/kitchen/info.txt"
    data_dir = "/home/haytam-elourrat/VISNAV/DATA/kitchen/data"
    
    color_files = sorted(glob.glob(os.path.join(data_dir, "*.color.jpg")))
    
    if end_idx is not None:
        color_files = color_files[:end_idx]
        
    if not color_files or start_idx >= len(color_files):
        print(f"Error: start_idx ({start_idx}) >= number of images ({len(color_files)}) or no images found in {data_dir}.")
        return None, None, None
        
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
        return None, None, None
    scene.load(mesh_path)
    
    trajectory = []
    print(f"Building trajectory with {len(color_files) - start_idx - 1} steps...")
    
    for i in range(start_idx + 1, len(color_files)):
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
            print(f"Warning: Target pose '{target_pose_path}' not found.")
            target_pose = None
            
        trajectory.append((real_target_img, target_pose))
        
    return scene, cam, trajectory


def main():
    parser = argparse.ArgumentParser(description="Run Mesh + ScanNet Trajectory")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index for the trajectory")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending index for the trajectory")
    parser.add_argument("--resume", type=str, help="Path to a previous run directory to resume from")
    args = parser.parse_args()

    # Determine start_target_idx and output_dir
    start_target_idx = 0
    resume_dir = args.resume

    if resume_dir:
        if not os.path.isdir(resume_dir):
            print(f"Error: Resume directory '{resume_dir}' does not exist.")
            return

        est_path = os.path.join(resume_dir, "trajectory_estimated.txt")
        if not os.path.exists(est_path):
            print(f"Error: Trajectory file '{est_path}' not found in resume directory.")
            return

        # Read last completed waypoint
        with open(est_path, "r") as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Trajectory file '{est_path}' is empty. Starting from target 0.")
            else:
                last_line = lines[-1].strip().split()
                # TUM format: timestamp tx ty tz qx qy qz qw
                last_ts = float(last_line[0])
                last_target_idx = int(round(last_ts))
                start_target_idx = last_target_idx + 1

                print(f"Resuming from target {start_target_idx + 1} based on {est_path}")

                # Extract last pose from TUM line to set camera
                t = np.array([float(last_line[1]), float(last_line[2]), float(last_line[3])])
                q = np.array([float(last_line[4]), float(last_line[5]), float(last_line[6]), float(last_line[7])])
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat(q).as_matrix()
                last_pose = np.eye(4)
                last_pose[:3, :3] = rot
                last_pose[:3, 3] = t

    scene, cam, trajectory = run_mesh_scannet_traj(start_idx=args.start_idx, end_idx=args.end_idx)

    if scene is None or cam is None or not trajectory:
        print("Failed to initialize setup.")
        return

    if resume_dir and 'last_pose' in locals():
        print("Setting camera to last known pose from trajectory...")
        cam.pose = last_pose

    controller = FBVSController(
        lambda_gain=1.5,
        max_velocity=1.0,
        use_moge=True,
        ratio=0
    )

    if resume_dir:
        output_dir = resume_dir
    else:
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
        start_target_idx=start_target_idx
    )

if __name__ == "__main__":
    main()
