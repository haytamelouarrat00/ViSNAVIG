import open3d as o3d
import numpy as np
import warnings
from .base_scene import BaseScene

class MeshScene(BaseScene):
    """Scene handler for 3D meshes using Open3D."""

    def __init__(self, width: int = 640, height: int = 480):
        self.mesh = None
        self.width = width
        self.height = height
        self.renderer = None

    def load(self, path: str) -> None:
        """Loads a 3D mesh from a file using Open3D."""
        self.mesh = o3d.io.read_triangle_mesh(path)
        if not self.mesh.has_triangles():
            raise ValueError(f"Failed to load mesh or mesh has no triangles: {path}")
        
        self.mesh.compute_vertex_normals()
        
        # Initialize the OffscreenRenderer
        try:
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            self.renderer.scene.add_geometry("mesh", self.mesh, material)
        except Exception as e:
            warnings.warn(f"Failed to initialize OffscreenRenderer (headless mode might require EGL): {e}")
            self.renderer = None

    def render(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the mesh from the given pose."""
        if self.mesh is None:
            raise RuntimeError("Scene not loaded. Call load() first.")
            
        if self.renderer is None:
            # Fallback for environments without GUI/EGL support
            warnings.warn("Renderer is not initialized. Returning a blank image.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Open3D expects extrinsics as World-to-Camera (T_wc)
        # Ensure matrices are contiguous float64 for Open3D C++ bridge
        intrinsics_64 = np.ascontiguousarray(intrinsics.astype(np.float64))
        pose_64 = np.ascontiguousarray(pose.astype(np.float64))
        
        # setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
        # Note: This has been observed to cause segmentation faults in some headless 
        # environments with missing EGL/OpenGL drivers.
        self.renderer.setup_camera(intrinsics_64, pose_64, self.width, self.height)
        
        img = self.renderer.render_to_image()
        return np.asarray(img)

    def render_depth(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the depth map of the mesh from the given pose."""
        if self.mesh is None:
            raise RuntimeError("Scene not loaded. Call load() first.")
            
        if self.renderer is None:
            # Fallback for environments without GUI/EGL support
            warnings.warn("Renderer is not initialized. Returning a blank depth image.")
            return np.zeros((self.height, self.width), dtype=np.float32)

        intrinsics_64 = np.ascontiguousarray(intrinsics.astype(np.float64))
        pose_64 = np.ascontiguousarray(pose.astype(np.float64))
        
        self.renderer.setup_camera(intrinsics_64, pose_64, self.width, self.height)
        
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)
        return np.asarray(depth_img, dtype=np.float32)
