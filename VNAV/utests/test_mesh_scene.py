import unittest
import numpy as np
import open3d as o3d
import os
from VNAV.scenes.mesh_scene import MeshScene

class TestMeshScene(unittest.TestCase):
    def setUp(self):
        self.scene = MeshScene(width=320, height=240)
        # Create a dummy mesh for testing
        self.test_mesh_path = "test_mesh.ply"
        mesh = o3d.geometry.TriangleMesh.create_box()
        o3d.io.write_triangle_mesh(self.test_mesh_path, mesh)

    def tearDown(self):
        if os.path.exists(self.test_mesh_path):
            os.remove(self.test_mesh_path)

    def test_load_mesh(self):
        self.scene.load(self.test_mesh_path)
        self.assertIsNotNone(self.scene.mesh)

    def test_render_mesh(self):
        self.scene.load(self.test_mesh_path)
        pose = np.eye(4)
        
        # Create a mock intrinsic matrix
        intrinsics = np.array([
            [250.0, 0.0, 160.0],
            [0.0, 250.0, 120.0],
            [0.0, 0.0, 1.0]
        ])
        
        img = self.scene.render(pose, intrinsics)
        self.assertEqual(img.shape, (240, 320, 3))

if __name__ == '__main__':
    unittest.main()
