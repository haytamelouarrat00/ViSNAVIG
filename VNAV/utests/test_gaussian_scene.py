import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from VNAV.scenes.gaussian_scene import GaussianScene

class TestGaussianScene(unittest.TestCase):
    def setUp(self):
        self.scene = GaussianScene(width=320, height=240, device="cpu")

    @patch('VNAV.scenes.gaussian_scene.PlyData')
    def test_load_gaussian(self, mock_plydata):
        # Mock the plydata structure
        mock_vertex = {
            'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]),
            'opacity': np.array([1.0]),
            'scale_0': np.array([1.0]), 'scale_1': np.array([1.0]), 'scale_2': np.array([1.0]),
            'rot_0': np.array([1.0]), 'rot_1': np.array([0.0]), 'rot_2': np.array([0.0]), 'rot_3': np.array([0.0]),
            'f_dc_0': np.array([0.5]), 'f_dc_1': np.array([0.5]), 'f_dc_2': np.array([0.5])
        }
        
        # Mock properties to simulate scale names
        class MockProperty:
            def __init__(self, name):
                self.name = name
                
        mock_vertex_obj = MagicMock()
        mock_vertex_obj.properties = [MockProperty('scale_0'), MockProperty('scale_1'), MockProperty('scale_2')]
        mock_vertex_obj.__getitem__.side_effect = mock_vertex.__getitem__
        
        mock_read = MagicMock()
        mock_read.__getitem__.return_value = mock_vertex_obj
        mock_plydata.read.return_value = mock_read
        
        self.scene.load("dummy_path.ply")
        self.assertIsNotNone(self.scene.means)
        self.assertEqual(len(self.scene.means), 1)

    @patch('VNAV.scenes.gaussian_scene.GaussianRasterizer')
    def test_render_gaussian(self, mock_rasterizer):
        # Set dummy tensors to bypass load
        self.scene.means = torch.zeros((1, 3))
        self.scene.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.scene.scales = torch.ones((1, 3))
        self.scene.opacities = torch.ones((1, 1))
        self.scene.shs = torch.zeros((1, 1, 3))
        
        # Mock rasterizer return
        mock_instance = MagicMock()
        # Return a 3-tuple (rendered_image, radii, depth_or_alpha)
        mock_instance.return_value = (torch.zeros((3, 240, 320)), torch.zeros(1), torch.ones((1, 240, 320)))
        mock_rasterizer.return_value = mock_instance

        pose = np.eye(4)
        intrinsics = np.eye(3)
        img = self.scene.render(pose, intrinsics)
        self.assertEqual(img.shape, (240, 320, 3))
        
        depth = self.scene.render_depth(pose, intrinsics)
        self.assertEqual(depth.shape, (240, 320))
        
    def test_render_without_load(self):
        with self.assertRaises(RuntimeError):
            self.scene.render(np.eye(4), np.eye(3))

if __name__ == '__main__':
    unittest.main()
