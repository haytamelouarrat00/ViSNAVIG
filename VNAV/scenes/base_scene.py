import abc
import numpy as np

class BaseScene(abc.ABC):
    """Abstract base class for all scene representations."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the scene from the specified path.
        
        Args:
            path (str): The path to the scene file or directory.
        """
        pass

    @abc.abstractmethod
    def render(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Renders an image from the given pose and camera intrinsics.
        
        Args:
            pose (np.ndarray): 4x4 numpy array representing the camera pose (extrinsics: world to camera).
            intrinsics (np.ndarray): 3x3 numpy array representing camera intrinsics.
            
        Returns:
            np.ndarray: The rendered image, typically as an HxWx3 uint8 numpy array.
        """
        pass

    @abc.abstractmethod
    def render_depth(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Renders a depth map from the given pose and camera intrinsics.
        
        Args:
            pose (np.ndarray): 4x4 numpy array representing the camera pose (extrinsics: world to camera).
            intrinsics (np.ndarray): 3x3 numpy array representing camera intrinsics.
            
        Returns:
            np.ndarray: The rendered depth map, typically as an HxW float32 numpy array.
        """
        pass
