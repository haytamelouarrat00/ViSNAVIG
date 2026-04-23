import numpy as np
from .base_controller import BaseController
from VNAV.features import XFeatMatcher
from VNAV.utilities import MoGe2DepthExtractor

class FBVSController(BaseController):
    """
    Feature-Based Visual Servoing (FBVS) Controller.
    Computes camera velocity by extracting features, matching them, and 
    inverting the image-based Interaction Matrix (Image Jacobian).
    """
    def __init__(self, lambda_gain: float = 1.0, max_velocity: float = 0.5, use_moge: bool = True, ratio: int = 1):
        """
        Args:
            lambda_gain (float): Exponential convergence rate.
            max_velocity (float): Maximum allowed velocity norm to prevent unstable jumps.
            use_moge (bool): If True, uses MoGe-2 to estimate depth from the RGB image 
                             instead of relying on the scene's perfect depth buffer.
            ratio (int): Determines when we redetect new features vs reprojecting them.
                         0 = detect once and reproject forever. N = redetect every N iterations. 1 = detect every iteration.
        """
        self.lambda_gain = lambda_gain
        self.max_velocity = max_velocity
        self.use_moge = use_moge
        self.ratio = ratio
        self.iteration_count = 0
        self.tracked_pts_target = None
        self.tracked_P_q_world = None
        self.current_error_norm = 0.0
        self.current_matches = (np.array([]), np.array([]))
        
        print(f"Initializing FBVS Controller (Ratio: {ratio})...")
        # 1. Default Feature Extractor: XFeat
        print("Loading XFeat Matcher...")
        self.matcher = XFeatMatcher(top_k=4096)
        
        # 2. Default Depth Estimator: MoGe-2
        if self.use_moge:
            self.depth_estimator = MoGe2DepthExtractor(variant="vits", half_precision=True)
        else:
            self.depth_estimator = None
            
    def reset(self):
        """Resets tracking state and iteration counter for a new target in the trajectory."""
        self.iteration_count = 0
        self.tracked_pts_target = None
        self.tracked_P_q_world = None
        self.current_matches = (np.array([]), np.array([]))
        if hasattr(self, 'target_depth'):
            self.target_depth = None

    def compute_velocity(self, current_image: np.ndarray, current_depth: np.ndarray, target_image: np.ndarray, intrinsics: np.ndarray, current_pose: np.ndarray = None, target_pose: np.ndarray = None) -> np.ndarray:
        should_match = (self.ratio == 1) or (self.ratio > 0 and self.iteration_count % self.ratio == 0) or (self.ratio == 0 and self.iteration_count == 0)
        
        if should_match:
            # 1. Extract and match features
            pts_curr, pts_target = self.matcher.match(current_image, target_image)
            
            if len(pts_curr) < 6:
                print(f"[FBVS] Warning: Not enough matches ({len(pts_curr)} < 6). Returning zero velocity.")
                self.current_matches = (np.array([]), np.array([]))
                self.iteration_count += 1
                return np.zeros(6, dtype=np.float64)

            # 1.5 Apply NeRF-IBVS Stage 1 Filter: Reprojection Distance
            if current_pose is not None and target_pose is not None and self.use_moge:
                if getattr(self, 'target_depth', None) is None:
                    print("[FBVS] Caching target depth for reprojection filter...")
                    self.target_depth = self.depth_estimator.get_depth(target_image)
                    
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                h_t, w_t = self.target_depth.shape
                
                P_q_world = []
                valid_mask = []
                
                for (u_t, v_t) in pts_target:
                    u_int = int(np.clip(round(u_t), 0, w_t - 1))
                    v_int = int(np.clip(round(v_t), 0, h_t - 1))
                    Z_t = self.target_depth[v_int, u_int]
                    
                    if Z_t <= 0.05 or not np.isfinite(Z_t):
                        valid_mask.append(False)
                        P_q_world.append([0, 0, 0])
                    else:
                        valid_mask.append(True)
                        X_c = (u_t - cx) * Z_t / fx
                        Y_c = (v_t - cy) * Z_t / fy
                        P_c = np.array([X_c, Y_c, Z_t, 1.0])
                        P_w = target_pose @ P_c
                        P_q_world.append(P_w[:3])
                
                valid_mask = np.array(valid_mask)
                if np.sum(valid_mask) > 0:
                    pts_curr_f = pts_curr[valid_mask]
                    pts_target_f = pts_target[valid_mask]
                    P_q_world_f = np.array(P_q_world)[valid_mask]
                    
                    from VNAV.features.filters import filter_by_reprojection_distance
                    T_wc = np.linalg.inv(current_pose)
                    pts_curr_filtered, pts_target_filtered, P_q_world_filtered, inliers = filter_by_reprojection_distance(
                        pts_curr_f, pts_target_f, P_q_world_f, T_wc, intrinsics, tau=float('inf')
                    )
                    
                    pts_curr = pts_curr_filtered
                    pts_target = pts_target_filtered
                    self.tracked_P_q_world = P_q_world_filtered
                else:
                    self.tracked_P_q_world = None
                    
            if len(pts_curr) < 6:
                print(f"[FBVS] Warning: Not enough matches after filter ({len(pts_curr)} < 6).")
                self.current_matches = (np.array([]), np.array([]))
                self.iteration_count += 1
                return np.zeros(6, dtype=np.float64)

            # 1.6 Apply NeRF-IBVS Stage 2 Filter: RANSAC for Geometric Consistency
            if len(pts_curr) >= 5:
                import cv2
                # findEssentialMat enforces epipolar geometry and handles non-collinear matching
                E, inliers = cv2.findEssentialMat(pts_curr, pts_target, cameraMatrix=intrinsics, method=cv2.RANSAC, prob=0.99, threshold=3.0)
                if inliers is not None:
                    inliers = inliers.ravel().astype(bool)
                    pts_curr = pts_curr[inliers]
                    pts_target = pts_target[inliers]
                    if getattr(self, 'tracked_P_q_world', None) is not None:
                        self.tracked_P_q_world = self.tracked_P_q_world[inliers]

            if len(pts_curr) < 6:
                print(f"[FBVS] Warning: Not enough matches after filter ({len(pts_curr)} < 6).")
                self.current_matches = (np.array([]), np.array([]))
                self.iteration_count += 1
                return np.zeros(6, dtype=np.float64)
                
            self.tracked_pts_target = pts_target
            
        else:
            # Reproject mode
            if getattr(self, 'tracked_pts_target', None) is None or getattr(self, 'tracked_P_q_world', None) is None or len(self.tracked_pts_target) < 6:
                print("[FBVS] Warning: No tracked points to reproject. Returning zero velocity.")
                self.current_matches = (np.array([]), np.array([]))
                self.iteration_count += 1
                return np.zeros(6, dtype=np.float64)
                
            # Project tracked_P_q_world into the current camera frame
            T_wc = np.linalg.inv(current_pose)
            N = self.tracked_P_q_world.shape[0]
            P_q_homo = np.hstack((self.tracked_P_q_world, np.ones((N, 1))))
            P_c = (T_wc @ P_q_homo.T).T  # Shape: (N, 4)
            
            valid_z_mask = P_c[:, 2] > 1e-5
            p_2d_homo = (intrinsics @ P_c[:, :3].T).T
            z_safe = np.where(valid_z_mask, P_c[:, 2], 1.0)
            n_r_hat = p_2d_homo[:, :2] / z_safe[:, None]
            
            # Filter out points that fall out of image bounds
            h, w = current_image.shape[:2]
            in_bounds_mask = (n_r_hat[:, 0] >= 0) & (n_r_hat[:, 0] < w) & (n_r_hat[:, 1] >= 0) & (n_r_hat[:, 1] < h)
            valid_mask = valid_z_mask & in_bounds_mask
            
            pts_curr = n_r_hat[valid_mask]
            pts_target = self.tracked_pts_target[valid_mask]
            
            # Update the tracked sets to only keep valid ones
            self.tracked_pts_target = pts_target
            self.tracked_P_q_world = self.tracked_P_q_world[valid_mask]
            
            if len(pts_curr) < 6:
                print(f"[FBVS] Warning: Not enough points after reprojection ({len(pts_curr)} < 6).")
                self.current_matches = (np.array([]), np.array([]))
                self.iteration_count += 1
                return np.zeros(6, dtype=np.float64)

        self.iteration_count += 1

        # 2. Retrieve Depth
        if self.use_moge:
            # Estimate metric depth from the current virtual RGB view
            depth_map = self.depth_estimator.get_depth(current_image)
        else:
            # Use the ground-truth depth buffer from the renderer
            depth_map = current_depth

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        L_e = []
        e = []
        valid_pts_c = []
        valid_pts_t = []

        # 3. Compute normalized coordinates and build Interaction Matrix
        h, w = depth_map.shape
        valid_points_count = 0

        for i in range(len(pts_curr)):
            u_c, v_c = pts_curr[i]
            u_t, v_t = pts_target[i]

            # Get depth at current pixel
            u_c_int = int(np.clip(round(u_c), 0, w - 1))
            v_c_int = int(np.clip(round(v_c), 0, h - 1))
            
            Z = depth_map[v_c_int, u_c_int]

            # Skip invalid depths
            if Z <= 0.05 or not np.isfinite(Z):
                continue 

            valid_pts_c.append((u_c, v_c))
            valid_pts_t.append((u_t, v_t))

            # Normalized image coordinates
            x_c = (u_c - cx) / fx
            y_c = (v_c - cy) / fy
            x_t = (u_t - cx) / fx
            y_t = (v_t - cy) / fy

            # 2D Error vector (Current - Target)
            e.append(x_c - x_t)
            e.append(y_c - y_t)

            # Interaction matrix L_e for a single 2D point (2x6)
            L_e_i = np.array([
                [-1.0/Z,    0.0,     x_c/Z,  x_c*y_c,      -(1.0 + x_c**2),  y_c ],
                [   0.0, -1.0/Z,     y_c/Z,  1.0 + y_c**2, -x_c*y_c,        -x_c ]
            ])
            L_e.append(L_e_i)
            valid_points_count += 1

        if valid_points_count < 6:
            print(f"[FBVS] Warning: Not enough valid depth points ({valid_points_count} < 6). Returning zero velocity.")
            self.current_matches = (np.array([]), np.array([]))
            return np.zeros(6, dtype=np.float64)

        # Convert to numpy arrays
        L_e = np.vstack(L_e)      # Shape: (2N, 6)
        e = np.array(e)           # Shape: (2N,)
        
        self.current_error_norm = np.linalg.norm(e)
        self.current_matches = (np.array(valid_pts_c), np.array(valid_pts_t))

        # 4. Compute control law: v_c = -lambda * pseudo_inverse(L_e) * e
        L_e_pinv = np.linalg.pinv(L_e)
        v_c = -self.lambda_gain * (L_e_pinv @ e)

        # 5. Clip velocity
        v_norm = np.linalg.norm(v_c)
        if v_norm > self.max_velocity:
            v_c = v_c * (self.max_velocity / v_norm)

        return v_c
