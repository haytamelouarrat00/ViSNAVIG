import numpy as np
import cv2
from .base_controller import BaseController

class DVSController(BaseController):
    """
    Direct Visual Servoing (DVS) Controller.
    Computes camera velocity by directly minimizing the photometric (intensity) error 
    between the current view and the target view.
    """
    def __init__(self, lambda_gain: float = 1.0):
        self.lambda_gain = lambda_gain
        
    def compute_velocity(self, current_image: np.ndarray, current_depth: np.ndarray, target_image: np.ndarray, intrinsics: np.ndarray, current_pose: np.ndarray = None, target_pose: np.ndarray = None) -> np.ndarray:
        # Convert images to grayscale and normalize to [0, 1]
        if current_image.ndim == 3:
            curr_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        else:
            curr_gray = current_image.astype(np.float64) / 255.0
            
        if target_image.ndim == 3:
            targ_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        else:
            targ_gray = target_image.astype(np.float64) / 255.0

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        H, W = curr_gray.shape
        
        # 1. Compute photometric error e = I_current - I_target
        e_full = (curr_gray - targ_gray)
        
        # 2. Compute spatial image gradients (dI/du, dI/dv)
        dI_dv, dI_du = np.gradient(curr_gray)
        
        # Scale to get Ix = dI/dx, Iy = dI/dy where x,y are normalized coordinates
        Ix_full = fx * dI_du
        Iy_full = fy * dI_dv
        
        # 3. Build dense Interaction Matrix L_e
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        x_full = (u - cx) / fx
        y_full = (v - cy) / fy
        
        # Create a mask to discard border pixels and invalid depths (similarly to ViSP)
        border = 10
        valid_mask = (current_depth > 0)
        valid_mask[:border, :] = False
        valid_mask[-border:, :] = False
        valid_mask[:, :border] = False
        valid_mask[:, -border:] = False
        
        # Flatten valid entries
        valid_flat = valid_mask.flatten()
        
        e = e_full.flatten()[valid_flat]
        Ix = Ix_full.flatten()[valid_flat]
        Iy = Iy_full.flatten()[valid_flat]
        x = x_full.flatten()[valid_flat]
        y = y_full.flatten()[valid_flat]
        Z_inv = 1.0 / current_depth.flatten()[valid_flat]
        
        L_e = np.zeros((len(e), 6), dtype=np.float64)
        L_e[:, 0] = Ix * Z_inv
        L_e[:, 1] = Iy * Z_inv
        L_e[:, 2] = -(x * Ix + y * Iy) * Z_inv
        L_e[:, 3] = -Ix * x * y - (1.0 + y**2) * Iy
        L_e[:, 4] = (1.0 + x**2) * Ix + Iy * x * y
        L_e[:, 5] = Iy * x - Ix * y
        
        # 4. Compute velocity: v_c = -lambda * pseudo_inverse(L_e) * e
        # To compute this efficiently: L_e^+ = (L_e^T * L_e)^-1 * L_e^T
        H_mat = L_e.T @ L_e
        J_err = L_e.T @ e
        
        try:
            v_c = -self.lambda_gain * np.linalg.solve(H_mat, J_err)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            H_pinv = np.linalg.pinv(H_mat)
            v_c = -self.lambda_gain * H_pinv @ J_err
            
        return v_c
