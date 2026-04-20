import os
import sys
import numpy as np
from typing import Tuple
from .base_matcher import BaseMatcher

# Safely point to the sibling VS folder
VS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../VS")
)

class XFeatMatcher(BaseMatcher):
    """
    Accelerated feature matcher using the XFeat architecture.
    It expects the module to be located in the sibling `VS` directory.
    """
    def __init__(self, top_k: int = 4096, min_cossim: float = -1):
        if VS_DIR not in sys.path:
            sys.path.append(VS_DIR)
            
        try:
            from accelerated_features.modules.xfeat import XFeat
        except ImportError as e:
            raise ImportError(
                f"Could not import XFeat from {VS_DIR}. "
                "Ensure the 'accelerated_features' module exists in the VS repository."
            ) from e
            
        self.top_k = top_k
        self.min_cossim = min_cossim
        self.xfeat = XFeat(top_k=top_k)

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # xfeat.match_xfeat inherently handles NumPy arrays of shape (H,W,C) or (H,W)
        pts1, pts2 = self.xfeat.match_xfeat(img1, img2, top_k=self.top_k, min_cossim=self.min_cossim)
        return pts1, pts2
