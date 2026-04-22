import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from VNAV.scenes import BaseScene
from VNAV.cameras import Camera
from VNAV.controllers.base_controller import BaseController

def visual_servoing_loop(
    scene: BaseScene,
    camera: Camera,
    target_image: np.ndarray,
    controller: BaseController,
    max_iterations: int = 200,
    dt: float = 0.05,
    error_tolerance: float = 0.2,
    velocity_epsilon: float = 1e-4,
    target_pose: np.ndarray = None
):
    """
    Executes the visual servoing control loop cleanly and minimalistically.
    
    Args:
        scene (BaseScene): The loaded 3D scene (Mesh or 3DGS).
        camera (Camera): The camera model initialized with intrinsics and an initial pose.
        target_image (np.ndarray): The real-world target image to servo towards.
        controller (BaseController): The logic module that outputs a velocity command.
        max_iterations (int): Maximum number of control loop iterations.
        dt (float): Time step for velocity integration.
        error_tolerance (float): Threshold for the error norm to stop the servoing.
        velocity_epsilon (float): Threshold for the velocity gradient (change) to stop the servoing.
        target_pose (np.ndarray): Ground truth target 4x4 pose (Camera-to-World) for displaying error.
    """
    print("Starting Visual Servoing Loop...")
    
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(14, 6))
    img_display = None
    
    prev_v_c = None
    
    for i in range(max_iterations):
        start_time = time.time()
        
        # 1. Render current virtual view and depth map
        rendered_img = camera.render(scene)
        rendered_depth = camera.render_depth(scene)
        
        # 2. Controller magic: compute velocity command based on the views
        v_c = controller.compute_velocity(
            rendered_img, rendered_depth, target_image, camera.K,
            current_pose=camera.pose, target_pose=target_pose
        )
        
        # 3. Apply control velocity to update camera pose
        camera.apply_velocity(v_c, dt=dt)
        
        # Calculate norms for stopping conditions and visualization
        v_norm = np.linalg.norm(v_c)
        e_norm = getattr(controller, 'current_error_norm', 0.0)
        
        # Check early stopping conditions
        stop_reason = None
        if e_norm > 0 and e_norm < error_tolerance:
            stop_reason = f"Error norm ({e_norm:.6f}) below tolerance ({error_tolerance})"
        elif prev_v_c is not None:
            # Gradient is the norm of the change in velocity divided by the time step
            v_c_grad = np.linalg.norm(v_c - prev_v_c) / dt
            if v_c_grad < velocity_epsilon:
                stop_reason = f"Velocity gradient ({v_c_grad:.6f}) below epsilon ({velocity_epsilon})"
                
        prev_v_c = v_c.copy()
        
        # 4. Visualization
        import cv2
        if rendered_img.shape != target_image.shape:
            target_image_resized = cv2.resize(target_image, (rendered_img.shape[1], rendered_img.shape[0]))
        else:
            target_image_resized = target_image

        combined_img = np.hstack((rendered_img, target_image_resized))
        
        # Add feature match visualization overlay
        if hasattr(controller, 'current_matches'):
            pts_curr, pts_target = controller.current_matches
            if len(pts_curr) > 0 and len(pts_target) > 0:
                overlay = np.zeros_like(combined_img)
                offset_x = rendered_img.shape[1]
                
                # Draw lines and points
                for (u_c, v_c), (u_t, v_t) in zip(pts_curr, pts_target):
                    pt1 = (int(round(u_c)), int(round(v_c)))
                    pt2 = (int(round(u_t)) + offset_x, int(round(v_t)))
                    
                    cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(overlay, pt1, 2, (0, 0, 255), -1)
                    cv2.circle(overlay, pt2, 2, (255, 0, 0), -1)
                
                # Blend with combined_img
                alpha = 0.4
                combined_img = cv2.addWeighted(overlay, alpha, combined_img, 1 - alpha, 0)
                
        if img_display is None:
            img_display = ax.imshow(combined_img)
            ax.axis('off')
        else:
            img_display.set_data(combined_img)
            
        fps = 1.0 / (time.time() - start_time + 1e-5)
        
        # Compute real-world ground truth error if target_pose is available
        pose_error_str = ""
        if target_pose is not None:
            current_pose = camera.pose
            
            # Position error (L2 distance)
            dist_m = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
            
            # Rotation error (angle in radians)
            try:
                curr_rot = R.from_matrix(current_pose[:3, :3])
                tgt_rot = R.from_matrix(target_pose[:3, :3])
                rel_rot = tgt_rot.inv() * curr_rot
                angle_rad = np.linalg.norm(rel_rot.as_rotvec())
                pose_error_str = f" | Dist: {dist_m:.3f} m | Angle: {angle_rad:.3f} rad"
            except Exception as e:
                pose_error_str = f" | Dist: {dist_m:.3f} m | Angle: Err"
        
        ax.set_title(
            f"Visual Servoing | Iter: {i+1}/{max_iterations} | FPS: {fps:.1f}\n"
            f"||v_c||: {v_norm:.4f} m/s | ||e||: {e_norm:.4f}{pose_error_str}\n"
            f"Left: Rendered (Moving) | Right: Target (Static)"
        )
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
        
        if stop_reason:
            print(f"\nTarget reached at iteration {i+1}!")
            print(f"Reason: {stop_reason}")
            if target_pose is not None:
                print(f"Final Distance: {dist_m:.4f} m, Final Angle: {angle_rad:.4f} rad")
            break
            
    plt.ioff()
    plt.show()
    print("Visual Servoing Loop Finished.")