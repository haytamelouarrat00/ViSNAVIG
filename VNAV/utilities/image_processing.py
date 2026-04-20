import cv2
import numpy as np

def get_edge_map(img: np.ndarray, low_threshold: int = 70, high_threshold: int = 100) -> np.ndarray:
    """
    Computes a binary edge map using Canny edge detection.
    
    Args:
        img (np.ndarray): Input image (RGB or Grayscale).
        low_threshold (int): Lower bound for hysteresis.
        high_threshold (int): Upper bound for hysteresis.
        
    Returns:
        np.ndarray: Binary edge map (uint8: 255 for edges, 0 otherwise).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Gaussian blur to reduce noise before Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges

def compute_chamfer_distance(edges1: np.ndarray, edges2: np.ndarray) -> float:
    """
    Computes the symmetric Chamfer distance between two binary edge maps.
    The distance is the average pixel distance from each edge point in edges1
    to the nearest edge point in edges2, and vice-versa.
    
    Args:
        edges1 (np.ndarray): First binary edge map (255 for edges).
        edges2 (np.ndarray): Second binary edge map (255 for edges).
        
    Returns:
        float: Symmetric Chamfer distance in pixels. Returns 0 if both are empty.
    """
    # Inverse the binary image: distanceTransform measures distance to 0 (non-edge)
    # So we want edges to be 0 and non-edges to be 255
    dist1 = cv2.distanceTransform(cv2.bitwise_not(edges1), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(cv2.bitwise_not(edges2), cv2.DIST_L2, 5)
    
    # Points in edge map 1
    pts1 = np.where(edges1 > 0)
    # Points in edge map 2
    pts2 = np.where(edges2 > 0)
    
    # Average distance from pts2 to nearest in edges1
    d2_to_1 = np.mean(dist1[pts2]) if len(pts2[0]) > 0 else 0.0
    # Average distance from pts1 to nearest in edges2
    d1_to_2 = np.mean(dist2[pts1]) if len(pts1[0]) > 0 else 0.0
    
    return (d1_to_2 + d2_to_1) / 2.0
