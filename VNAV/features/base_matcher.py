import abc
import numpy as np
from typing import Tuple

class BaseMatcher(abc.ABC):
    """Abstract base class for all feature detection and matching architectures."""
    
    @abc.abstractmethod
    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features from two images and finds correspondences.
        
        Args:
            img1 (np.ndarray): First image as HxWx3 (RGB) or HxW (Grayscale).
            img2 (np.ndarray): Second image as HxWx3 (RGB) or HxW (Grayscale).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two N x 2 numpy arrays containing the 
                                           (x, y) coordinates of the matches in img1 and img2.
        """
        pass
