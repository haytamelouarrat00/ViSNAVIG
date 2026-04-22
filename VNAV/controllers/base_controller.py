import abc
import numpy as np

class BaseController(abc.ABC):
    """Abstract base class for Visual Servoing controllers."""
    
    @abc.abstractmethod
    def compute_velocity(self, current_image: np.ndarray, current_depth: np.ndarray, target_image: np.ndarray, intrinsics: np.ndarray, current_pose: np.ndarray = None, target_pose: np.ndarray = None) -> np.ndarray:
        """
        Computes the 6-DoF camera velocity command.
        
        Args:
            current_image (np.ndarray): The current rendered RGB or Grayscale image.
            current_depth (np.ndarray): The current rendered depth map.
            target_image (np.ndarray): The target RGB or Grayscale image.
            intrinsics (np.ndarray): The 3x3 camera intrinsics matrix.
            current_pose (np.ndarray, optional): The current 4x4 camera pose (Camera-to-World).
            target_pose (np.ndarray, optional): The target 4x4 camera pose (Camera-to-World).
            
        Returns:
            np.ndarray: A 6-element velocity vector [vx, vy, vz, wx, wy, wz].
        """
        pass
