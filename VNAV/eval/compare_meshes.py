#!/usr/bin/env python3
"""
Mesh Comparison Script for 3D Reconstructions
Evaluates multiple mesh models using:
1. Pairwise agreement matrix (Chamfer distance)
2. Intrinsic mesh quality metrics
3. Photometric reprojection consistency (using COLMAP poses)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("Warning: tabulate not installed. Falling back to basic printing. Install with: pip install tabulate")
    def tabulate(df, headers, tablefmt, showindex=True):
        return str(df)

try:
    import cv2
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Warning: cv2 or scikit-image not installed. Reprojection metrics will be limited.")
    cv2, psnr, ssim = None, None, None

def load_meshes(mesh_paths):
    print(f"Loading {len(mesh_paths)} meshes...")
    meshes = {}
    for path in mesh_paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
            
        name = os.path.basename(path).split('.')[0]
        # In case of duplicate names
        base_name = name
        counter = 1
        while name in meshes:
            name = f"{base_name}_{counter}"
            counter += 1
            
        print(f"  - {name} ({path})")
        meshes[name] = {
            'path': path,
            'o3d': o3d.io.read_triangle_mesh(path),
            'trimesh': trimesh.load(path, process=False, force='mesh')
        }
    return meshes

def compute_pairwise_distances(meshes, num_samples=100000):
    """Computes pairwise Chamfer distances between a list of meshes."""
    print(f"\n--- 1. Computing Pairwise Agreement Matrix (Chamfer Distance) ---")
    names = list(meshes.keys())
    n = len(names)
    
    if n < 2:
        print("Need at least 2 meshes for pairwise comparison.")
        return None
        
    pcds = {}
    for name, mesh_data in meshes.items():
        print(f"Sampling {num_samples} points from {name}...")
        try:
            # Sample uniformly from the surface
            pcds[name] = mesh_data['o3d'].sample_points_uniformly(number_of_points=num_samples)
        except Exception as e:
            print(f"Error sampling {name}: {e}")
            return None
    
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            pcd_i = pcds[names[i]]
            pcd_j = pcds[names[j]]
            
            # Compute point-to-point distances
            dist_i_to_j = np.asarray(pcd_i.compute_point_cloud_distance(pcd_j))
            dist_j_to_i = np.asarray(pcd_j.compute_point_cloud_distance(pcd_i))
            
            # Symmetric Chamfer distance
            chamfer_dist = np.mean(dist_i_to_j) + np.mean(dist_j_to_i)
            
            dist_matrix[i, j] = chamfer_dist
            dist_matrix[j, i] = chamfer_dist
            
    df = pd.DataFrame(dist_matrix, index=names, columns=names)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("\nInterpretation: Methods that agree with known-reliable methods (like BundleFusion) or have low distances with each other are more likely to be geometrically accurate.")
    return df

def compute_intrinsic_metrics(meshes):
    """Computes intrinsic geometric quality metrics for each mesh."""
    print(f"\n--- 2. Computing Intrinsic Mesh Quality Metrics ---")
    results = []
    
    for name, mesh_data in meshes.items():
        tm = mesh_data['trimesh']
        
        try:
            edges_lengths = tm.edges_unique_length
            mean_edge_length = np.mean(edges_lengths)
            std_edge_length = np.std(edges_lengths)
        except Exception:
            mean_edge_length = float('nan')
            std_edge_length = float('nan')
            
        # Basic bounds and density
        extents = tm.extents
        diag_len = np.linalg.norm(extents)
            
        metrics = {
            'Method': name,
            'Vertices': len(tm.vertices),
            'Faces': len(tm.faces),
            'Surface Area': tm.area,
            'Components': tm.body_count,
            'Euler Char.': tm.euler_number,
            'Watertight': tm.is_watertight,
            'Non-manifold Edges': len(tm.nonmanifold_edges) if hasattr(tm, 'nonmanifold_edges') else 'N/A',
            'Mean Edge Length': f"{mean_edge_length:.4f} ± {std_edge_length:.4f}"
        }
        
        # Triangle quality checking (aspect ratio approximation)
        try:
            # Face angles are computed on demand in trimesh
            min_angles = np.min(tm.face_angles, axis=1)
            metrics['Degenerate Faces (%)'] = f"{(np.sum(min_angles < 0.05) / len(tm.faces) * 100):.2f}%"
        except Exception:
            metrics['Degenerate Faces (%)'] = 'N/A'
            
        results.append(metrics)
        
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    return df

def read_colmap_cameras(cameras_file):
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if not parts: continue
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras

def read_colmap_images(images_file):
    images = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#'):
                i += 1
                continue
            parts = line.strip().split()
            if not parts:
                i += 1
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = [tx, ty, tz]
            
            images[image_id] = {'name': name, 'camera_id': camera_id, 'extrinsics': extrinsics}
            i += 2 # Skip points2D line
    return images

def compute_reprojection_consistency(meshes, colmap_dir, images_dir):
    """Computes Photometric Reprojection Consistency using Open3D OffscreenRenderer."""
    print(f"\n--- 3. Computing Photometric Reprojection Consistency ---")
    if not colmap_dir or not images_dir:
        print("Info: '--colmap_dir' or '--images_dir' not provided.")
        return None
        
    if cv2 is None or psnr is None:
        print("Error: cv2 and scikit-image are required for this step.")
        print("Install with: pip install opencv-python scikit-image")
        return None

    cameras_file = os.path.join(colmap_dir, 'cameras.txt')
    images_file = os.path.join(colmap_dir, 'images.txt')
    
    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        print(f"Warning: Could not find cameras.txt or images.txt in {colmap_dir}")
        return None

    try:
        cameras = read_colmap_cameras(cameras_file)
        colmap_images = read_colmap_images(images_file)
    except Exception as e:
        print(f"Error parsing COLMAP files: {e}")
        return None

    if not colmap_images:
        print("No images found in COLMAP data.")
        return None

    print(f"✓ Found {len(colmap_images)} images and {len(cameras)} cameras in COLMAP.")

    results = []
    
    # Evaluate on a subset to save time if there are many images
    eval_images = list(colmap_images.values())
    max_eval_images = 10 
    if len(eval_images) > max_eval_images:
        indices = np.linspace(0, len(eval_images) - 1, max_eval_images, dtype=int)
        eval_images = [eval_images[i] for i in indices]
        print(f"Evaluating on a representative subset of {max_eval_images} images...")

    for name, mesh_data in meshes.items():
        print(f"Rendering {name}...")
        mesh = mesh_data['o3d']
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Determine material to use
        material = o3d.visualization.rendering.MaterialRecord()
        if not mesh.has_vertex_colors() and not mesh.has_textures():
            material.shader = "defaultLit"
        else:
            material.shader = "defaultUnlit"

        mesh_psnrs = []
        mesh_ssims = []

        cam_id = eval_images[0]['camera_id']
        width = cameras[cam_id]['width']
        height = cameras[cam_id]['height']
        
        try:
            render = o3d.visualization.rendering.OffscreenRenderer(width, height)
            render.scene.add_geometry("mesh", mesh, material)
            render.scene.set_background([0.0, 0.0, 0.0, 1.0]) # Black background
        except Exception as e:
            print(f"Error initializing Open3D OffscreenRenderer: {e}")
            print("You might need a display or headless rendering setup (e.g. xvfb-run).")
            return None

        for img_info in eval_images:
            img_name = img_info['name']
            gt_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(gt_path):
                print(f"  Warning: GT image not found: {gt_path}")
                continue
                
            gt_img = cv2.imread(gt_path)
            if gt_img is None:
                continue
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            cam = cameras[img_info['camera_id']]
            
            # Setup camera intrinsics
            # Handle different camera models generically where params[0,1] are fx, fy, params[2,3] are cx, cy usually
            # For SIMPLE_RADIAL, params are f, cx, cy, k
            if cam['model'] == 'SIMPLE_RADIAL':
                fx, fy = cam['params'][0], cam['params'][0]
                cx, cy = cam['params'][1], cam['params'][2]
            elif cam['model'] == 'PINHOLE':
                fx, fy = cam['params'][0], cam['params'][1]
                cx, cy = cam['params'][2], cam['params'][3]
            else:
                # Fallback assuming standard order
                fx = cam['params'][0]
                fy = cam['params'][1] if len(cam['params']) > 2 else cam['params'][0]
                cx = cam['params'][2] if len(cam['params']) > 2 else cam['params'][1]
                cy = cam['params'][3] if len(cam['params']) > 2 else cam['params'][2]

            intrinsics = o3d.camera.PinholeCameraIntrinsic(cam['width'], cam['height'], fx, fy, cx, cy)
            
            render.setup_camera(intrinsics, img_info['extrinsics'])
            
            rendered_img = np.asarray(render.render_to_image())
            
            if gt_img.shape != rendered_img.shape:
                rendered_img = cv2.resize(rendered_img, (gt_img.shape[1], gt_img.shape[0]))
                
            p = psnr(gt_img, rendered_img)
            
            min_dim = min(gt_img.shape[0], gt_img.shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0: win_size -= 1
            if win_size < 3: win_size = 3
            
            try:
                s = ssim(gt_img, rendered_img, channel_axis=-1, data_range=255, win_size=win_size)
            except Exception:
                s = 0.0
            
            mesh_psnrs.append(p)
            mesh_ssims.append(s)
            
        render.scene.remove_geometry("mesh")
        del render
        
        results.append({
            'Method': name,
            'Render PSNR': f"{np.mean(mesh_psnrs):.2f}" if mesh_psnrs else "N/A",
            'Render SSIM': f"{np.mean(mesh_ssims):.4f}" if mesh_ssims else "N/A"
        })
        
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation script to compare 3D mesh models without absolute ground truth."
    )
    parser.add_argument('--meshes', nargs='+', required=True, 
                        help="List of paths to .ply or .obj files to compare.")
    parser.add_argument('--colmap_dir', type=str, default=None, 
                        help="Path to COLMAP directory containing cameras.txt and images.txt")
    parser.add_argument('--images_dir', type=str, default=None, 
                        help="Path to directory containing the original input images")
    parser.add_argument('--output_dir', type=str, default='eval_results', 
                        help="Directory to save the CSV reports")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    meshes = load_meshes(args.meshes)
    
    # 1. Pairwise Agreement Matrix
    dist_df = compute_pairwise_distances(meshes)
    if dist_df is not None:
        dist_df.to_csv(os.path.join(args.output_dir, "pairwise_chamfer_distances.csv"))
    
    # 2. Intrinsic Mesh Quality Metrics
    intrinsic_df = compute_intrinsic_metrics(meshes)
    if intrinsic_df is not None:
        intrinsic_df.to_csv(os.path.join(args.output_dir, "intrinsic_metrics.csv"), index=False)
    
    # 3. Photometric Reprojection Consistency
    reproj_df = compute_reprojection_consistency(meshes, args.colmap_dir, args.images_dir)
    if reproj_df is not None:
        reproj_df.to_csv(os.path.join(args.output_dir, "reprojection_metrics.csv"), index=False)
        
    print(f"\nAll reports saved to '{args.output_dir}/'")

if __name__ == "__main__":
    main()
