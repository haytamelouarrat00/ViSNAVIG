import cv2
import numpy as np
from typing import Tuple
from .base_matcher import BaseMatcher

class SIFTMatcher(BaseMatcher):
    """
    Classic feature matcher using OpenCV's SIFT detector and FlannBasedMatcher.
    """
    def __init__(self, max_features: int = 4096, ratio_threshold: float = 0.75):
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        
        # FLANN parameters for SIFT (algorithm 1 is KDTree)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.ratio_threshold = ratio_threshold

    def match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # SIFT expects grayscale or 8-bit images
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return np.empty((0, 2)), np.empty((0, 2))

        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        if not good_matches:
            return np.empty((0, 2)), np.empty((0, 2))

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2
