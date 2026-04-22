import argparse
import os
import sys
import numpy as np
import pycolmap

# Ensure the parent directory is in the Python path to allow `import VNAV.*`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cameras.camera import Camera

def get_scannet_pose(pose_path):
    """Loads a 4x4 pose matrix from a ScanNet pose.txt file."""
    return np.loadtxt(pose_path)

def calibrate_colmap(colmap_dir, scannet_dir, output_dir=None):
    """
    Computes the scale factor between a COLMAP reconstruction and ScanNet ground-truth poses,
    and applies this scale to the entire COLMAP reconstruction to permanently convert it to metric meters.
    """
    if output_dir is None:
        output_dir = colmap_dir # Overwrite by default
        
    print(f"Loading COLMAP reconstruction from {colmap_dir}...")
    if not os.path.exists(os.path.join(colmap_dir, "cameras.bin")):
         print(f"Error: Could not find COLMAP binary files in {colmap_dir}")
         return
         
    reconstruction = pycolmap.Reconstruction(colmap_dir)
    
    # Sort images by name to ensure sequentiality
    images = sorted(reconstruction.images.values(), key=lambda img: img.name)
    
    if len(images) < 2:
        print("Error: Not enough images in COLMAP reconstruction to calibrate.")
        return
        
    # Find two images that both exist in ScanNet and have a non-zero translation
    # to compute a robust scale factor. We compare the first image with subsequent ones
    # until we find a sufficient physical distance.
    base_img = images[0]
    
    # ScanNet poses usually share the exact same prefix as the color image
    base_pose_name = base_img.name.replace(".color.jpg", ".pose.txt").replace(".jpg", ".txt")
    base_pose_path = os.path.join(scannet_dir, base_pose_name)
    
    if not os.path.exists(base_pose_path):
        print(f"Error: Could not find ScanNet pose for {base_img.name} at {base_pose_path}.")
        return
        
    scannet_pose_base = get_scannet_pose(base_pose_path)
    
    colmap_cameras_path = os.path.join(colmap_dir, "cameras.bin")
    colmap_images_path = os.path.join(colmap_dir, "images.bin")
    
    cam_base = Camera.from_colmap(colmap_cameras_path, camera_id=base_img.camera_id)
    try:
        cam_base.set_pose_from_colmap(colmap_images_path, image_name=base_img.name)
        colmap_pose_base = cam_base.pose
    except ValueError as e:
        print(f"Error extracting base pose: {e}")
        return

    scale_factor = None
    
    print("Searching for a suitable frame pair to compute scale...")
    for i in range(1, len(images)):
        target_img = images[i]
        target_pose_name = target_img.name.replace(".color.jpg", ".pose.txt").replace(".jpg", ".txt")
        target_pose_path = os.path.join(scannet_dir, target_pose_name)
        
        if not os.path.exists(target_pose_path):
            continue
            
        scannet_pose_target = get_scannet_pose(target_pose_path)
        # Calculate metric distance (Euclidean norm of translation difference)
        scannet_dist = np.linalg.norm(scannet_pose_base[:3, 3] - scannet_pose_target[:3, 3])
        
        cam_target = Camera.from_colmap(colmap_cameras_path, camera_id=target_img.camera_id)
        try:
            cam_target.set_pose_from_colmap(colmap_images_path, image_name=target_img.name)
            colmap_pose_target = cam_target.pose
        except ValueError:
            continue
            
        colmap_dist = np.linalg.norm(colmap_pose_base[:3, 3] - colmap_pose_target[:3, 3])
        
        # Require at least 5cm of physical movement to avoid dividing by tiny, noisy numbers
        if scannet_dist > 0.05 and colmap_dist > 1e-5:
            scale_factor = scannet_dist / colmap_dist
            print(f"Success! Found sufficient baseline between '{base_img.name}' and '{target_img.name}'")
            print(f"  ScanNet (Metric) Distance: {scannet_dist:.6f} meters")
            print(f"  COLMAP (Arbitrary) Distance: {colmap_dist:.6f} units")
            print(f"  Calculated Scale Factor: {scale_factor:.6f}")
            break
                
    if scale_factor is None:
        print("Error: Could not find a pair of images with sufficient translation (>5cm) to compute scale reliably.")
        return
        
    print(f"\nApplying scale transformation ({scale_factor:.6f}) to entire COLMAP reconstruction...")
    # pycolmap.Sim3d(scale, rotation, translation)
    # Scales the scene geometry (poses and 3D points) while keeping the origin and orientation identical.
    sim3d = pycolmap.Sim3d(scale_factor, pycolmap.Rotation3d(), np.zeros(3))
    reconstruction.transform(sim3d)
    
    os.makedirs(output_dir, exist_ok=True)
    # Write the calibrated cameras.bin, images.bin, and points3D.bin back to disk
    reconstruction.write(output_dir)
    print(f"✅ Calibrated reconstruction successfully saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate COLMAP reconstruction scale to True Metric using ScanNet poses.")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Directory containing COLMAP sparse model (cameras.bin, images.bin, points3D.bin)")
    parser.add_argument("--scannet_dir", type=str, required=True, help="Directory containing ScanNet .pose.txt files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save calibrated reconstruction (overwrites colmap_dir by default)")
    
    args = parser.parse_args()
    calibrate_colmap(args.colmap_dir, args.scannet_dir, args.output_dir)
