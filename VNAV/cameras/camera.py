import numpy as np
from scipy.spatial.transform import Rotation as R

class Camera:
    """
    A robust camera model for visual servoing and 3D rendering.
    Handles intrinsics, 6 DoF pose (extrinsics), velocity-based movement,
    and 3D-to-2D projection.
    """
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float, pose: np.ndarray = None):
        """
        Initializes the camera.
        
        Args:
            width (int): Image width.
            height (int): Image height.
            fx (float): Focal length x.
            fy (float): Focal length y.
            cx (float): Principal point x.
            cy (float): Principal point y.
            pose (np.ndarray, optional): 4x4 Camera-to-World transformation matrix. Defaults to Identity.
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        if pose is None:
            self.T_cw = np.eye(4, dtype=np.float64)
        else:
            self.pose = pose

    @classmethod
    def from_colmap(cls, cameras_bin_path: str, camera_id: int = 1) -> "Camera":
        """
        Creates a Camera instance from a COLMAP cameras.bin file.
        Extracts intrinsics (ignoring distortion for now).
        
        Args:
            cameras_bin_path (str): Path to cameras.bin
            camera_id (int): ID of the camera to load. Defaults to 1.
            
        Returns:
            Camera: Initialized camera with identity pose.
        """
        import struct
        import os
        
        if not os.path.exists(cameras_bin_path):
            raise FileNotFoundError(f"cameras.bin not found at {cameras_bin_path}")
            
        with open(cameras_bin_path, "rb") as fid:
            # 64-bit unsigned int for number of cameras
            num_cameras = struct.unpack("<Q", fid.read(8))[0]
            
            for _ in range(num_cameras):
                cam_id = struct.unpack("<i", fid.read(4))[0]
                model_id = struct.unpack("<i", fid.read(4))[0]
                width = struct.unpack("<Q", fid.read(8))[0]
                height = struct.unpack("<Q", fid.read(8))[0]
                
                # Model param counts based on COLMAP source
                model_to_num_params = {
                    0: 3,   # SIMPLE_PINHOLE
                    1: 4,   # PINHOLE
                    2: 4,   # SIMPLE_RADIAL
                    3: 5,   # RADIAL
                    4: 8,   # OPENCV
                    5: 8,   # OPENCV_FISHEYE
                    6: 12,  # FULL_OPENCV
                    7: 3,   # SIMPLE_RADIAL_FISHEYE
                    8: 4,   # RADIAL_FISHEYE
                    9: 5,   # THIN_PRISM_FISHEYE
                }
                
                if model_id not in model_to_num_params:
                    raise ValueError(f"Unknown camera model ID: {model_id} in {cameras_bin_path}")
                    
                num_params = model_to_num_params[model_id]
                params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
                
                if cam_id == camera_id:
                    # Extract fx, fy, cx, cy based on model
                    # Models where fx == fy: SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE
                    if model_id in [0, 2, 3, 7, 8]:
                        fx = fy = params[0]
                        cx = params[1]
                        cy = params[2]
                    else:
                        fx = params[0]
                        fy = params[1]
                        cx = params[2]
                        cy = params[3]
                        
                    return cls(int(width), int(height), float(fx), float(fy), float(cx), float(cy))
                    
        raise ValueError(f"Camera ID {camera_id} not found in {cameras_bin_path}")

    @classmethod
    def from_dataset_info(cls, info_path: str, sensor_type: str = "color") -> "Camera":
        """
        Creates a Camera instance from a dataset info file (e.g., BundleFusion info.txt format).
        
        Args:
            info_path (str): Path to the info file.
            sensor_type (str): 'color' or 'depth' to specify which camera to load. Defaults to 'color'.
            
        Returns:
            Camera: Initialized camera with identity pose.
        """
        import os
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Info file not found at {info_path}")
            
        params = {}
        with open(info_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.split('=', 1)
                    params[key.strip()] = value.strip()
                    
        prefix = "m_color" if sensor_type == "color" else "m_depth"
        width_key = f"{prefix}Width"
        height_key = f"{prefix}Height"
        intrinsic_key = f"m_calibration{sensor_type.capitalize()}Intrinsic"
        
        if width_key not in params or height_key not in params or intrinsic_key not in params:
            raise ValueError(f"Missing required parameters for {sensor_type} sensor in {info_path}")
            
        width = int(params[width_key])
        height = int(params[height_key])
        
        # Parse the 4x4 intrinsic matrix values (row-major)
        intrinsics_vals = [float(v) for v in params[intrinsic_key].split()]
        if len(intrinsics_vals) != 16:
            raise ValueError(f"Expected 16 values for intrinsic matrix, got {len(intrinsics_vals)}")
            
        # The matrix is stored row-major:
        # [0: fx, 1: 0,  2: cx, 3: 0]
        # [4: 0,  5: fy, 6: cy, 7: 0]
        # [8: 0,  9: 0, 10: 1, 11: 0]
        # [12:0, 13: 0, 14: 0, 15: 1]
        fx = intrinsics_vals[0]
        cx = intrinsics_vals[2]
        fy = intrinsics_vals[5]
        cy = intrinsics_vals[6]
        
        return cls(width, height, fx, fy, cx, cy)

    @property
    def pose(self) -> np.ndarray:
        """Returns the camera-to-world transformation matrix (4x4)."""
        return self.T_cw.copy()

    @pose.setter
    def pose(self, new_pose: np.ndarray):
        """Sets the camera-to-world transformation matrix (4x4)."""
        new_pose = np.asarray(new_pose, dtype=np.float64)
        if new_pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix")
        self.T_cw = new_pose

    @property
    def extrinsics(self) -> np.ndarray:
        """Returns the world-to-camera transformation matrix (4x4)."""
        return np.linalg.inv(self.T_cw)

    def set_pose_from_scannet(self, pose_path: str) -> None:
        """
        Loads and sets the camera pose from a ScanNet-style .txt file.
        These files contain a 4x4 Camera-to-World (T_cw) transformation matrix.
        
        Args:
            pose_path (str): Path to the pose.txt file.
        """
        import os
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found at {pose_path}")
            
        pose_matrix = np.loadtxt(pose_path)
        if pose_matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 matrix, but got {pose_matrix.shape} in {pose_path}")
            
        self.pose = pose_matrix

    def set_pose_from_colmap(self, images_bin_path: str, image_id: int = None, image_name: str = None) -> None:
        """
        Loads and sets the camera pose from a COLMAP images.bin file.
        COLMAP stores World-to-Camera (T_wc) as a quaternion (qw, qx, qy, qz) and translation (tx, ty, tz).
        This method computes T_wc and inverts it to set the Camera-to-World (T_cw) pose.
        
        Args:
            images_bin_path (str): Path to images.bin.
            image_id (int, optional): The ID of the image to find.
            image_name (str, optional): The name of the image to find.
        """
        import struct
        import os
        from scipy.spatial.transform import Rotation as R
        
        if not os.path.exists(images_bin_path):
            raise FileNotFoundError(f"images.bin not found at {images_bin_path}")
            
        if image_id is None and image_name is None:
            raise ValueError("Must provide either image_id or image_name to locate the pose.")
            
        with open(images_bin_path, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            
            for _ in range(num_reg_images):
                img_id = struct.unpack("<i", fid.read(4))[0]
                q_and_t = struct.unpack("<7d", fid.read(56)) # qw, qx, qy, qz, tx, ty, tz
                cam_id = struct.unpack("<i", fid.read(4))[0]
                
                name_bytes = b""
                while True:
                    char = fid.read(1)
                    if char == b"\x00":
                        break
                    name_bytes += char
                name = name_bytes.decode("utf-8")
                
                num_points2d = struct.unpack("<Q", fid.read(8))[0]
                # Skip 2D points data (X, Y, POINT3D_ID) -> 8 + 8 + 8 = 24 bytes per point
                fid.seek(num_points2d * 24, os.SEEK_CUR)
                
                if (image_id is not None and img_id == image_id) or (image_name is not None and name == image_name):
                    qw, qx, qy, qz, tx, ty, tz = q_and_t
                    
                    # Scipy expects scalar-last quaternion format: [x, y, z, w]
                    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    
                    T_wc = np.eye(4, dtype=np.float64)
                    T_wc[:3, :3] = rot
                    T_wc[:3, 3] = [tx, ty, tz]
                    
                    # Invert T_wc to get T_cw and set it
                    self.pose = np.linalg.inv(T_wc)
                    return
                    
        raise ValueError(f"Image with id={image_id} or name={image_name} not found in {images_bin_path}")

    def apply_velocity(self, velocity: np.ndarray, dt: float):
        """
        Updates the camera pose given a 6-DoF velocity command in the camera's local frame.
        
        Args:
            velocity (np.ndarray): [vx, vy, vz, wx, wy, wz] (linear velocity, angular velocity).
            dt (float): Time step.
        """
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != (6,):
            raise ValueError("Velocity must be a 6-element vector: [vx, vy, vz, wx, wy, wz]")
            
        v = velocity[:3] * dt
        w = velocity[3:] * dt
        
        theta = np.linalg.norm(w)
        T_delta = np.eye(4, dtype=np.float64)
        
        if theta < 1e-8:
            T_delta[:3, 3] = v
        else:
            w_cross = np.array([
                [0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]
            ], dtype=np.float64)
            
            # Rotation
            T_delta[:3, :3] = R.from_rotvec(w).as_matrix()
            
            # Translation using the SE(3) exponential map
            I = np.eye(3, dtype=np.float64)
            V = I + ((1 - np.cos(theta)) / (theta**2)) * w_cross + ((theta - np.sin(theta)) / (theta**3)) * (w_cross @ w_cross)
            T_delta[:3, 3] = V @ v
            
        self.T_cw = self.T_cw @ T_delta

    def project(self, points_3d: np.ndarray):
        """
        Projects 3D points in the world frame into 2D pixel coordinates.
        
        Args:
            points_3d (np.ndarray): Nx3 array of 3D points.
            
        Returns:
            tuple: (Nx2 array of 2D pixel coordinates, boolean mask of valid points in front of the camera)
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d[None, :]
            
        N = points_3d.shape[0]
        pts_homo = np.hstack((points_3d, np.ones((N, 1))))
        
        T_wc = self.extrinsics
        pts_cam = (T_wc @ pts_homo.T).T  # Nx4
        
        z = pts_cam[:, 2]
        valid_mask = z > 1e-5
        
        pts_2d_homo = (self.K @ pts_cam[:, :3].T).T # Nx3
        
        # Avoid division by zero for invalid points
        z_safe = np.where(valid_mask, z, 1.0)
        u = pts_2d_homo[:, 0] / z_safe
        v = pts_2d_homo[:, 1] / z_safe
        
        uv = np.column_stack((u, v))
        return uv, valid_mask
        
    def render(self, scene):
        """
        Renders the current view of the scene from the camera's perspective.
        
        Args:
            scene (BaseScene): The scene object to render.
            
        Returns:
            np.ndarray: Rendered image.
        """
        return scene.render(self.extrinsics, self.K)

    def render_depth(self, scene):
        """
        Renders the current depth map of the scene from the camera's perspective.
        
        Args:
            scene (BaseScene): The scene object to render.
            
        Returns:
            np.ndarray: Rendered depth map.
        """
        return scene.render_depth(self.extrinsics, self.K)
