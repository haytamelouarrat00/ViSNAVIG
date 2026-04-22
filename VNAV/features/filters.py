import numpy as np

def filter_by_reprojection_distance(
    pts_rendered: np.ndarray, 
    pts_query: np.ndarray, 
    P_q_world: np.ndarray, 
    T_wc: np.ndarray, 
    K: np.ndarray, 
    tau: float = 200.0
):
    """
    Filters feature matches based on the 3D-to-2D reprojection distance as described in NeRF-IBVS.
    
    Args:
        pts_rendered (np.ndarray): Nx2 array of 2D matched points in the rendered image (current view).
        pts_query (np.ndarray): Nx2 array of 2D matched points in the query image (target view).
        P_q_world (np.ndarray): Nx3 array of 3D world coordinates for the query points.
        T_wc (np.ndarray): 4x4 World-to-Camera extrinsic matrix of the current rendered view.
        K (np.ndarray): 3x3 camera intrinsics matrix.
        tau (float): Distance threshold in pixels.
        
    Returns:
        Tuple containing filtered arrays: (pts_rendered, pts_query, P_q_world, inlier_mask).
    """
    if len(P_q_world) == 0:
        return pts_rendered, pts_query, P_q_world, np.array([], dtype=bool)

    N = P_q_world.shape[0]
    # Step 1 & 2: Project 3D query coordinates into the rendered image plane
    P_q_homo = np.hstack((P_q_world, np.ones((N, 1))))
    
    # Transform from World to Camera frame: P_c = T_wc * P_q
    P_c = (T_wc @ P_q_homo.T).T  # Shape: (N, 4)
    
    # Keep only points that are strictly in front of the camera
    valid_z_mask = P_c[:, 2] > 1e-5
    
    # Project 3D camera coordinates to 2D pixel coordinates (n_r_hat)
    p_2d_homo = (K @ P_c[:, :3].T).T # Shape: (N, 3)
    
    # Divide by Z to get standard 2D pixel coordinates (u, v)
    z_safe = np.where(valid_z_mask, P_c[:, 2], 1.0) # Avoid division by zero
    n_r_hat = p_2d_homo[:, :2] / z_safe[:, None]
    
    # Step 3: Compute the L2 reprojection distance (d = || n_r_hat - n_r ||_2)
    d = np.linalg.norm(n_r_hat - pts_rendered, axis=1)
    
    # Step 4: Threshold filtering (d < tau)
    inlier_mask = (d < tau) & valid_z_mask
    
    return pts_rendered[inlier_mask], pts_query[inlier_mask], P_q_world[inlier_mask], inlier_mask