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
    def __init__(self, lambda_gain: float = 1.0, max_velocity: float = 0.5, use_moge: bool = True):
        """
        Args:
            lambda_gain (float): Exponential convergence rate.
            max_velocity (float): Maximum allowed velocity norm to prevent unstable jumps.
            use_moge (bool): If True, uses MoGe-2 to estimate depth from the RGB image 
                             instead of relying on the scene's perfect depth buffer.
        """
        self.lambda_gain = lambda_gain
        self.max_velocity = max_velocity
        self.use_moge = use_moge
        self.current_error_norm = 0.0
        
        print("Initializing FBVS Controller...")
        # 1. Default Feature Extractor: XFeat
        print("Loading XFeat Matcher...")
        self.matcher = XFeatMatcher(top_k=4096)
        
        # 2. Default Depth Estimator: MoGe-2
        if self.use_moge:
            self.depth_estimator = MoGe2DepthExtractor(variant="vits", half_precision=True)
        else:
            self.depth_estimator = None
            
    def compute_velocity(self, current_image: np.ndarray, current_depth: np.ndarray, target_image: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        # 1. Extract and match features
        pts_curr, pts_target = self.matcher.match(current_image, target_image)
        
        if len(pts_curr) < 6:
            print(f"[FBVS] Warning: Not enough matches ({len(pts_curr)} < 6). Returning zero velocity.")
            return np.zeros(6, dtype=np.float64)

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
            return np.zeros(6, dtype=np.float64)

        # Convert to numpy arrays
        L_e = np.vstack(L_e)      # Shape: (2N, 6)
        e = np.array(e)           # Shape: (2N,)
        
        self.current_error_norm = np.linalg.norm(e)

        # 4. Compute control law: v_c = -lambda * pseudo_inverse(L_e) * e
        L_e_pinv = np.linalg.pinv(L_e)
        v_c = -self.lambda_gain * (L_e_pinv @ e)

        # 5. Clip velocity
        v_norm = np.linalg.norm(v_c)
        if v_norm > self.max_velocity:
            v_c = v_c * (self.max_velocity / v_norm)

        return v_c
