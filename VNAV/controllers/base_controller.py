import abc
import numpy as np

class BaseController(abc.ABC):
    """Abstract base class for Visual Servoing controllers."""
    
    @abc.abstractmethod
    def compute_velocity(self, current_image: np.ndarray, current_depth: np.ndarray, target_image: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Computes the 6-DoF camera velocity command.
        
        Args:
            current_image (np.ndarray): The current rendered RGB or Grayscale image.
            current_depth (np.ndarray): The current rendered depth map.
            target_image (np.ndarray): The target RGB or Grayscale image.
            intrinsics (np.ndarray): The 3x3 camera intrinsics matrix.
            
        Returns:
            np.ndarray: A 6-element velocity vector [vx, vy, vz, wx, wy, wz].
        """
        pass
