import os
# Enforce headless rendering via EGL to avoid X11 errors on servers
os.environ['PYOPENGL_PLATFORM'] = 'egl' 

import numpy as np
import warnings
import pyrender
import trimesh
from .base_scene import BaseScene

class MeshScene(BaseScene):
    """Scene handler for 3D meshes using Pyrender (robust headless EGL rendering)."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.mesh = None
        self.scene = None
        self.mesh_node = None
        self.renderer = None

    def load(self, path: str) -> None:
        """Loads a 3D mesh from a file using Trimesh and sets up Pyrender."""
        try:
            # Load mesh with trimesh
            tm = trimesh.load(path)
            self.mesh = tm
            
            # Create pyrender mesh
            mesh = pyrender.Mesh.from_trimesh(tm)
            
            # Initialize scene with moderate ambient light to avoid washing out colors
            self.scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
            
            # Add mesh to scene
            self.mesh_node = self.scene.add(mesh)
            
            # Initialize renderer
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
            
            print(f"Loaded Mesh from {path} into Pyrender.")
        except Exception as e:
            warnings.warn(f"Failed to initialize Pyrender OffscreenRenderer: {e}")
            self.scene = None
            self.renderer = None

    def _run_render(self, pose: np.ndarray, intrinsics: np.ndarray):
        if self.scene is None or self.renderer is None:
            return None, None

        # The input `pose` is the camera's extrinsic matrix (World-to-Camera, T_wc)
        # Pyrender node poses expect Camera-to-World (T_cw)
        T_cw = np.linalg.inv(pose)
        
        # Furthermore, Pyrender uses the OpenGL camera convention (Y up, Z back)
        # Our extrinsics use the OpenCV camera convention (Y down, Z forward)
        # We apply a 180-degree rotation around the X-axis to convert the coordinates
        cv2gl = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float64)
        
        T_cw_gl = T_cw @ cv2gl

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        camera_node = self.scene.add(camera, pose=T_cw_gl)
        
        # Add a directional light attached to the camera (headlamp) with soft intensity
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        light_node = self.scene.add(light, pose=T_cw_gl)
        
        # Render the scene
        color, depth = self.renderer.render(self.scene)
        
        # Clean up the camera and light nodes for the next frame
        self.scene.remove_node(camera_node)
        self.scene.remove_node(light_node)
        
        return color, depth

    def render(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the mesh from the given pose."""
        if self.scene is None:
            raise RuntimeError("Scene not loaded. Call load() first.")
            
        color, depth = self._run_render(pose, intrinsics)
        
        if color is None:
            warnings.warn("Renderer failed. Returning a blank image.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        return color.copy()

    def render_depth(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the depth map of the mesh from the given pose."""
        if self.scene is None:
            raise RuntimeError("Scene not loaded. Call load() first.")
            
        color, depth = self._run_render(pose, intrinsics)
        
        if depth is None:
            warnings.warn("Renderer failed. Returning a blank depth image.")
            return np.zeros((self.height, self.width), dtype=np.float32)
            
        return depth.astype(np.float32).copy()
