import numpy as np
import time
import matplotlib.pyplot as plt
from VNAV.scenes import BaseScene
from VNAV.cameras import Camera
from VNAV.controllers.base_controller import BaseController

def visual_servoing_loop(
    scene: BaseScene,
    camera: Camera,
    target_image: np.ndarray,
    controller: BaseController,
    max_iterations: int = 200,
    dt: float = 0.05
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
    """
    print("Starting Visual Servoing Loop...")
    
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(14, 6))
    img_display = None
    
    for i in range(max_iterations):
        start_time = time.time()
        
        # 1. Render current virtual view and depth map
        rendered_img = camera.render(scene)
        rendered_depth = camera.render_depth(scene)
        
        # 2. Controller magic: compute velocity command based on the views
        v_c = controller.compute_velocity(rendered_img, rendered_depth, target_image, camera.K)
        
        # 3. Apply control velocity to update camera pose
        camera.apply_velocity(v_c, dt=dt)
        
        # 4. Visualization
        import cv2
        if rendered_img.shape != target_image.shape:
            target_image_resized = cv2.resize(target_image, (rendered_img.shape[1], rendered_img.shape[0]))
        else:
            target_image_resized = target_image

        combined_img = np.hstack((rendered_img, target_image_resized))
        
        if img_display is None:
            img_display = ax.imshow(combined_img)
            ax.axis('off')
        else:
            img_display.set_data(combined_img)
            
        fps = 1.0 / (time.time() - start_time + 1e-5)
        
        v_norm = np.linalg.norm(v_c)
        e_norm = getattr(controller, 'current_error_norm', 0.0)
        
        ax.set_title(
            f"Visual Servoing | Iter: {i+1}/{max_iterations} | FPS: {fps:.1f}\n"
            f"||v_c||: {v_norm:.4f} m/s | ||e||: {e_norm:.4f}\n"
            f"Left: Rendered (Moving) | Right: Target (Static)"
        )
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
            
    plt.ioff()
    plt.show()
    print("Visual Servoing Loop Finished.")
