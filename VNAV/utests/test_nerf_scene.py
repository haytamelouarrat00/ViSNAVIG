import unittest
import numpy as np
from VNAV.scenes.nerf_scene import NerfScene

class TestNerfScene(unittest.TestCase):
    def setUp(self):
        self.scene = NerfScene(width=320, height=240)

    def test_load_nerf(self):
        self.scene.load("dummy_path")
        self.assertIsNotNone(self.scene.model)

    def test_render_nerf(self):
        self.scene.load("dummy_path")
        pose = np.eye(4)
        intrinsics = np.eye(3)
        img = self.scene.render(pose, intrinsics)
        self.assertEqual(img.shape, (240, 320, 3))

    def test_render_without_load(self):
        with self.assertRaises(RuntimeError):
            self.scene.render(np.eye(4), np.eye(3))

if __name__ == '__main__':
    unittest.main()
