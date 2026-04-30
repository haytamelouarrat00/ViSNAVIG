import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cameras import Camera
from scenes import GaussianScene
# You can also import MeshScene depending on user input

def load_tum_trajectory(file_path):
    """
    Loads a TUM trajectory file.
    Format: timestamp tx ty tz qx qy qz qw
    Returns: list of (timestamp, 4x4 transform matrix)
    """
    poses = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
                
            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            
            pose = np.eye(4)
            pose[:3, 3] = [tx, ty, tz]
            pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            
            poses.append((ts, pose))
    return poses

def render_trajectory_frames(
    scene_type: str,
    scene_path: str,
    trajectory_file: str,
    camera_intrinsics_path: str,
    output_dir: str,
    camera_id: int = 1,
    resolution: tuple = (480, 640)
):
    """
    Given a scene and a TUM trajectory file, renders and saves the frames.
    """
    print(f"Loading {scene_type} scene from {scene_path}...")
    if scene_type == "gaussian":
        scene = GaussianScene(scene_path)
    else:
        # Add MeshScene if needed later
        raise ValueError(f"Unsupported scene type: {scene_type}")
        
    print(f"Loading camera parameters from {camera_intrinsics_path} (camera_id={camera_id})...")
    # Load camera from colmap
    colmap_cameras_path = os.path.join(camera_intrinsics_path, "cameras.bin")
    camera = Camera.from_colmap(colmap_cameras_path, camera_id=camera_id)
    # We might need to override resolution if user specified it differently, but colmap camera already has width/height.
    print(f"Camera resolution: {camera.width}x{camera.height}")

    print(f"Loading trajectory from {trajectory_file}...")
    trajectory = load_tum_trajectory(trajectory_file)
    print(f"Loaded {len(trajectory)} poses.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering frames to {output_dir}...")
    for idx, (ts, pose) in enumerate(trajectory):
        camera.pose = pose
        
        # Render the scene
        color, depth = scene.render(camera)
        
        # The color image from render is typically RGB float [0, 1] or uint8 [0, 255].
        # Assuming it's uint8 [0, 255] based on typical visual servoing outputs.
        if color.dtype != np.uint8:
            color = (np.clip(color, 0, 1) * 255).astype(np.uint8)
            
        img = Image.fromarray(color)
        
        # Format filename to be consistent with servoing.py output (e.g., 00000_render.jpg)
        filename = f"{idx:05d}_render.jpg"
        filepath = os.path.join(output_dir, filename)
        
        img.save(filepath, quality=90)
        
        if (idx + 1) % 10 == 0:
            print(f"Rendered {idx + 1}/{len(trajectory)} frames...")
            
    print(f"Done! Rendered {len(trajectory)} frames to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline rendering of a TUM trajectory using a VNAV Scene.")
    parser.add_argument("--scene_type", type=str, default="gaussian", choices=["gaussian", "mesh"], help="Type of the scene (gaussian or mesh).")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene model (e.g., .ply file).")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to the TUM trajectory file (e.g., trajectory_estimated.txt).")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Directory containing COLMAP cameras.bin for camera intrinsics.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the rendered frames.")
    
    args = parser.parse_args()
    
    render_trajectory_frames(
        scene_type=args.scene_type,
        scene_path=args.scene_path,
        trajectory_file=args.trajectory,
        camera_intrinsics_path=args.colmap_dir,
        output_dir=args.output_dir
    )
