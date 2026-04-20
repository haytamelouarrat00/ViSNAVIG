import unittest
import numpy as np
from VNAV.cameras.camera import Camera

class MockScene:
    def render(self, extrinsics, intrinsics):
        return (extrinsics, intrinsics)
        
    def render_depth(self, extrinsics, intrinsics):
        return (extrinsics, intrinsics, "depth")

class TestCamera(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.height = 480
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera = Camera(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

    def test_initialization(self):
        self.assertEqual(self.camera.width, self.width)
        self.assertEqual(self.camera.height, self.height)
        self.assertTrue(np.allclose(self.camera.K, np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ])))
        self.assertTrue(np.allclose(self.camera.pose, np.eye(4)))
        self.assertTrue(np.allclose(self.camera.extrinsics, np.eye(4)))

    def test_pose_setting(self):
        new_pose = np.eye(4)
        new_pose[0, 3] = 10.0
        self.camera.pose = new_pose
        self.assertTrue(np.allclose(self.camera.pose, new_pose))
        
        # Extrinsics should be the inverse
        expected_extrinsics = np.eye(4)
        expected_extrinsics[0, 3] = -10.0
        self.assertTrue(np.allclose(self.camera.extrinsics, expected_extrinsics))

        with self.assertRaises(ValueError):
            self.camera.pose = np.eye(3)

    def test_apply_velocity_translation(self):
        # 1 unit/sec in local X
        velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.camera.apply_velocity(velocity, dt=2.0)
        
        expected_pose = np.eye(4)
        expected_pose[0, 3] = 2.0
        self.assertTrue(np.allclose(self.camera.pose, expected_pose))

    def test_apply_velocity_rotation(self):
        # 90 degrees/sec around local Z
        velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2])
        self.camera.apply_velocity(velocity, dt=1.0)
        
        expected_pose = np.eye(4)
        expected_pose[:3, :3] = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        self.assertTrue(np.allclose(self.camera.pose, expected_pose, atol=1e-7))

    def test_apply_velocity_mixed(self):
        # Constant rotation and translation
        velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi/2])
        self.camera.apply_velocity(velocity, dt=1.0)
        
        # For a full 90 degree turn with v_x=1, we trace a quarter circle.
        # Just check that SE(3) exp map doesn't crash and modifies pose.
        new_pose = self.camera.pose
        self.assertFalse(np.allclose(new_pose, np.eye(4)))

    def test_project_point(self):
        # Point exactly 5 units in front of the camera
        pt_3d = np.array([0.0, 0.0, 5.0])
        uv, mask = self.camera.project(pt_3d)
        
        self.assertTrue(mask[0])
        self.assertTrue(np.allclose(uv[0], [self.cx, self.cy]))
        
        # Point behind the camera
        pt_3d_behind = np.array([0.0, 0.0, -5.0])
        uv, mask = self.camera.project(pt_3d_behind)
        self.assertFalse(mask[0])

    def test_project_with_pose(self):
        # Move camera 10 units in X
        new_pose = np.eye(4)
        new_pose[0, 3] = 10.0
        self.camera.pose = new_pose
        
        # Point at [10, 0, 5] in world -> [0, 0, 5] in camera
        pt_3d = np.array([10.0, 0.0, 5.0])
        uv, mask = self.camera.project(pt_3d)
        
        self.assertTrue(mask[0])
        self.assertTrue(np.allclose(uv[0], [self.cx, self.cy]))

    def test_render(self):
        scene = MockScene()
        pose, intrinsics = self.camera.render(scene)
        self.assertTrue(np.allclose(pose, self.camera.extrinsics))
        self.assertTrue(np.allclose(intrinsics, self.camera.K))
        
    def test_render_depth(self):
        scene = MockScene()
        pose, intrinsics, label = self.camera.render_depth(scene)
        self.assertTrue(np.allclose(pose, self.camera.extrinsics))
        self.assertTrue(np.allclose(intrinsics, self.camera.K))
        self.assertEqual(label, "depth")

    def test_from_colmap(self):
        import struct
        import os
        
        dummy_path = "test_cameras.bin"
        try:
            with open(dummy_path, "wb") as f:
                # num_cameras = 2
                f.write(struct.pack("<Q", 2))
                
                # Camera 1: PINHOLE (1)
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<Q", 800))
                f.write(struct.pack("<Q", 600))
                f.write(struct.pack("<4d", 500.0, 510.0, 400.0, 300.0))
                
                # Camera 2: SIMPLE_RADIAL (2)
                f.write(struct.pack("<i", 2))
                f.write(struct.pack("<i", 2))
                f.write(struct.pack("<Q", 1024))
                f.write(struct.pack("<Q", 768))
                f.write(struct.pack("<4d", 600.0, 512.0, 384.0, 0.1))
                
            # Load Camera 1
            cam1 = Camera.from_colmap(dummy_path, camera_id=1)
            self.assertEqual(cam1.width, 800)
            self.assertEqual(cam1.height, 600)
            self.assertEqual(cam1.fx, 500.0)
            self.assertEqual(cam1.fy, 510.0)
            self.assertEqual(cam1.cx, 400.0)
            self.assertEqual(cam1.cy, 300.0)
            
            # Load Camera 2
            cam2 = Camera.from_colmap(dummy_path, camera_id=2)
            self.assertEqual(cam2.width, 1024)
            self.assertEqual(cam2.height, 768)
            self.assertEqual(cam2.fx, 600.0)
            self.assertEqual(cam2.fy, 600.0)
            self.assertEqual(cam2.cx, 512.0)
            self.assertEqual(cam2.cy, 384.0)
            
            # Non-existent camera
            with self.assertRaises(ValueError):
                Camera.from_colmap(dummy_path, camera_id=3)
                
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_from_dataset_info(self):
        import os
        
        dummy_path = "test_info.txt"
        info_content = """m_versionNumber = 2
m_sensorName = StructureSensor
m_colorWidth = 1296
m_colorHeight = 968
m_depthWidth = 640
m_depthHeight = 480
m_depthShift = 1000
m_calibrationColorIntrinsic = 1158.3 0 649 0 0 1153.53 483.5 0 0 0 1 0 0 0 0 1 
m_calibrationColorExtrinsic = 1 0 0 -0 0 1 0 -0 0 0 1 -0 0 0 0 1 
m_calibrationDepthIntrinsic = 572 0 320 0 0 572 240 0 0 0 1 0 0 0 0 1 
m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
m_frames.size = 1101
"""
        try:
            with open(dummy_path, "w") as f:
                f.write(info_content)
                
            # Load Color Camera
            cam_color = Camera.from_dataset_info(dummy_path, sensor_type="color")
            self.assertEqual(cam_color.width, 1296)
            self.assertEqual(cam_color.height, 968)
            self.assertAlmostEqual(cam_color.fx, 1158.3)
            self.assertAlmostEqual(cam_color.fy, 1153.53)
            self.assertAlmostEqual(cam_color.cx, 649.0)
            self.assertAlmostEqual(cam_color.cy, 483.5)
            
            # Load Depth Camera
            cam_depth = Camera.from_dataset_info(dummy_path, sensor_type="depth")
            self.assertEqual(cam_depth.width, 640)
            self.assertEqual(cam_depth.height, 480)
            self.assertAlmostEqual(cam_depth.fx, 572.0)
            self.assertAlmostEqual(cam_depth.fy, 572.0)
            self.assertAlmostEqual(cam_depth.cx, 320.0)
            self.assertAlmostEqual(cam_depth.cy, 240.0)
            
            # Invalid sensor type
            with self.assertRaises(ValueError):
                Camera.from_dataset_info(dummy_path, sensor_type="infrared")
                
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_set_pose_from_scannet(self):
        import os
        dummy_path = "test_pose.txt"
        # Dummy T_cw
        pose_content = """-0.001512 0.301897 -0.95334 -2.068
0.998037 0.060151 0.017465 -0.650432
0.062617 -0.951441 -0.301396 2.01739
0 0 0 1
"""
        try:
            with open(dummy_path, "w") as f:
                f.write(pose_content)
                
            self.camera.set_pose_from_scannet(dummy_path)
            
            # Verify pose matrix
            expected_pose = np.array([
                [-0.001512, 0.301897, -0.95334, -2.068],
                [0.998037, 0.060151, 0.017465, -0.650432],
                [0.062617, -0.951441, -0.301396, 2.01739],
                [0, 0, 0, 1]
            ])
            self.assertTrue(np.allclose(self.camera.pose, expected_pose))
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_set_pose_from_colmap(self):
        import os
        import struct
        from scipy.spatial.transform import Rotation as R
        
        dummy_path = "test_images.bin"
        try:
            # We want to encode a specific T_wc to check if it parses and inverts correctly
            # Let's create T_wc: pure translation of +5 in X, and 90 deg rotation around Y
            r = R.from_euler('y', 90, degrees=True)
            qx, qy, qz, qw = r.as_quat() # scalar last
            tx, ty, tz = 5.0, 0.0, 0.0
            
            # The expected T_cw should be T_wc^-1
            T_wc = np.eye(4)
            T_wc[:3, :3] = r.as_matrix()
            T_wc[:3, 3] = [tx, ty, tz]
            expected_T_cw = np.linalg.inv(T_wc)
            
            with open(dummy_path, "wb") as f:
                # num_reg_images = 1
                f.write(struct.pack("<Q", 1))
                
                # Image 1
                f.write(struct.pack("<i", 42)) # image_id = 42
                f.write(struct.pack("<7d", qw, qx, qy, qz, tx, ty, tz))
                f.write(struct.pack("<i", 1)) # camera_id = 1
                
                # name "frame_42.png" + \0
                f.write(b"frame_42.png\x00")
                
                # 2 points2d
                f.write(struct.pack("<Q", 2))
                # point 1: X, Y, POINT3D_ID
                f.write(struct.pack("<ddq", 10.0, 20.0, 100))
                # point 2: X, Y, POINT3D_ID
                f.write(struct.pack("<ddq", 30.0, 40.0, 101))
                
            # Load by ID
            self.camera.set_pose_from_colmap(dummy_path, image_id=42)
            self.assertTrue(np.allclose(self.camera.pose, expected_T_cw))
            
            # Reset and load by Name
            self.camera.pose = np.eye(4)
            self.camera.set_pose_from_colmap(dummy_path, image_name="frame_42.png")
            self.assertTrue(np.allclose(self.camera.pose, expected_T_cw))
            
            # Test not found
            with self.assertRaises(ValueError):
                self.camera.set_pose_from_colmap(dummy_path, image_id=99)
                
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

if __name__ == '__main__':
    unittest.main()
