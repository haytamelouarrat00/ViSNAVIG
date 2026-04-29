"""Direct (photometric) Visual Servoing controller."""

from __future__ import annotations

import cv2
import numpy as np

from .base_controller import BaseController


class DVSController(BaseController):
    """
    Direct (Photometric) Visual Servoing (DVS) Controller.
    Inspired by ViSP's vpServoLuminance. Computes the camera velocity directly from
    image intensities (luminance) and depth without extracting geometric features.
    """

    def __init__(self, lambda_gain: float = 1.0, *args, **kwargs) -> None:
        """
        Args:
            lambda_gain (float): Control gain.
        """
        self.lambda_gain = lambda_gain
        self.current_error_norm: float = 0.0
        self.current_error_image: np.ndarray | None = None

    def reset(self) -> None:
        """Resets the controller state."""
        self.current_error_norm = 0.0
        self.current_error_image = None

    def compute_velocity(
        self,
        current_image: np.ndarray,
        current_depth: np.ndarray,
        target_image: np.ndarray,
        intrinsics: np.ndarray,
        current_pose: np.ndarray | None = None,
        target_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the 6-DoF camera velocity command using photometric visual servoing.
        """
        # Convert to grayscale if necessary
        if current_image.ndim == 3:
            curr_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_image

        if target_image.ndim == 3:
            targ_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            targ_gray = target_image

        curr_gray = curr_gray.astype(np.float32)
        targ_gray = targ_gray.astype(np.float32)

        # Compute photometric error e = I - I*
        error_image = curr_gray - targ_gray
        self.current_error_image = error_image
        self.current_error_norm = float(np.linalg.norm(error_image))

        # Compute image gradients (central differences)
        # grad_v is dI/dv (along rows/y-axis), grad_u is dI/du (along cols/x-axis)
        grad_v, grad_u = np.gradient(curr_gray)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        H, W = curr_gray.shape
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

        # Normalized coordinates
        x = (u_coords - cx) / fx
        y = (v_coords - cy) / fy

        Z = current_depth

        # Use only valid depth pixels to avoid division by zero or negative depths
        valid_mask = Z > 0.01

        # Flatten arrays for valid pixels
        x_f = x[valid_mask]
        y_f = y[valid_mask]
        Z_f = Z[valid_mask]
        grad_u_f = grad_u[valid_mask]
        grad_v_f = grad_v[valid_mask]
        e_f = error_image[valid_mask]

        # Convert image gradients to normalized coordinate gradients as in ViSP
        # Ix = px * dI/du, Iy = py * dI/dv
        Ix = grad_u_f * fx
        Iy = grad_v_f * fy

        # Formulate Interaction Matrix L_I (N x 6)
        L_vx = Ix / Z_f
        L_vy = Iy / Z_f
        L_vz = -(x_f * Ix + y_f * Iy) / Z_f
        L_wx = -x_f * y_f * Ix - (1 + y_f**2) * Iy
        L_wy = (1 + x_f**2) * Ix + x_f * y_f * Iy
        L_wz = x_f * Iy - y_f * Ix

        L = np.column_stack((L_vx, L_vy, L_vz, L_wx, L_wy, L_wz))

        # Control law: v = -lambda * L^+ * e
        # We solve L * v = -lambda * e
        v, _, _, _ = np.linalg.lstsq(L, -self.lambda_gain * e_f, rcond=None)

        return v.astype(np.float32)
