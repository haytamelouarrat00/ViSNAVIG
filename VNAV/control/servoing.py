import os
import queue
import threading
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from VNAV.scenes import BaseScene
from VNAV.cameras import Camera
from VNAV.controllers.base_controller import BaseController


# -----------------------------------------------------------------------------
# Output helpers (frame dump, TUM trajectory, evo evaluation)
# -----------------------------------------------------------------------------

def _pose_to_tum_line(timestamp: float, T_cw: np.ndarray) -> str:
    """Serialize a 4x4 camera-to-world pose to a TUM-format line.

    TUM format: ``timestamp tx ty tz qx qy qz qw`` (quaternion is scalar-last).
    """
    t = T_cw[:3, 3]
    q = R.from_matrix(T_cw[:3, :3]).as_quat()  # [x, y, z, w]
    return (
        f"{timestamp:.6f} "
        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
        f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
    )


def _save_tum_trajectory(path: str, timestamps, poses) -> None:
    """Write a list of camera-to-world poses to ``path`` in TUM format."""
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            f.write(_pose_to_tum_line(ts, T))


class _AsyncFrameWriter:
    """Background JPEG/PNG frame writer.

    Encode + disk IO runs in a worker thread so the servoing loop doesn't wait on it.
    A bounded queue applies natural backpressure if disk becomes the bottleneck
    (the main thread will block on put() instead of accumulating unbounded memory).

    NOTE: The ``rendered_img`` passed to ``submit()`` must not be mutated by the
    caller afterwards. In this codebase ``scene.render()`` returns a fresh array
    each iteration, so no copy is necessary.
    """

    _SENTINEL = object()

    def __init__(self, fmt: str = "jpg", quality: int = 90, max_queue: int = 8):
        self.fmt = fmt.lower()
        self.quality = int(quality)
        self.q = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._SENTINEL:
                break
            path, rendered_img, target_resized = item
            try:
                combined = np.hstack((rendered_img, target_resized))
                bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                if self.fmt in ("jpg", "jpeg"):
                    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                else:
                    cv2.imwrite(path, bgr)
            except Exception as e:
                # Printed on its own line so it doesn't collide with \r status.
                print(f"\n[frame writer] failed for {path}: {e}")

    def submit(self, path: str, rendered_img: np.ndarray, target_resized: np.ndarray):
        self.q.put((path, rendered_img, target_resized))

    def close(self):
        self.q.put(self._SENTINEL)
        self._thread.join()


def _resize_target(target_image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return target resized to (h, w) if needed, else the original array."""
    if target_image.shape[:2] == (h, w):
        return target_image
    return cv2.resize(target_image, (w, h))


def _run_evo_evaluation(gt_path: str, est_path: str, output_dir: str) -> None:
    """Read the saved trajectories back and render an evo comparison + APE plot."""
    try:
        from evo.tools import file_interface, plot
        from evo.core import metrics, sync
        from evo.core.metrics import PoseRelation
    except ImportError as e:
        print(f"[evo] Not installed; skipping trajectory evaluation: {e}")
        return

    try:
        traj_ref = file_interface.read_tum_trajectory_file(gt_path)
        traj_est = file_interface.read_tum_trajectory_file(est_path)
    except Exception as e:
        print(f"[evo] Failed to load trajectories: {e}")
        return

    if traj_ref.num_poses == 0 or traj_est.num_poses == 0:
        print("[evo] One of the trajectories is empty; skipping evaluation.")
        return

    try:
        traj_ref_sync, traj_est_sync = sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.01
        )
    except Exception as e:
        print(f"[evo] Timestamp association failed ({e}); falling back to raw trajectories.")
        traj_ref_sync, traj_est_sync = traj_ref, traj_est

    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref_sync, traj_est_sync))
    ape_stats = ape_metric.get_all_statistics()

    print("\n[evo] APE (translation part, meters):")
    for k, v in ape_stats.items():
        print(f"  {k}: {v:.6f}")

    plot_mode = plot.PlotMode.xyz

    # Both trajectories overlaid in 3D.
    fig_cmp = plt.figure(figsize=(10, 8))
    ax_cmp = plot.prepare_axis(fig_cmp, plot_mode)
    plot.traj(ax_cmp, plot_mode, traj_ref_sync, "--", "gray", "Ground Truth")
    plot.traj(ax_cmp, plot_mode, traj_est_sync, "-", "blue", "Estimated")
    ax_cmp.legend()
    ax_cmp.set_title(
        f"Trajectory Comparison | APE (m) mean={ape_stats.get('mean', 0.0):.4f} "
        f"rmse={ape_stats.get('rmse', 0.0):.4f}"
    )
    cmp_path = os.path.join(output_dir, "trajectory_comparison.png")
    fig_cmp.savefig(cmp_path, dpi=150, bbox_inches="tight")
    print(f"[evo] Saved comparison plot: {cmp_path}")

    # Estimated trajectory colored by APE error.
    fig_ape = plt.figure(figsize=(10, 8))
    ax_ape = plot.prepare_axis(fig_ape, plot_mode)
    plot.traj_colormap(
        ax_ape,
        traj_est_sync,
        ape_metric.error,
        plot_mode,
        min_map=float(ape_stats.get("min", 0.0)),
        max_map=float(ape_stats.get("max", 1.0)),
    )
    ax_ape.set_title("APE (translation) along Estimated Trajectory")
    ape_plot_path = os.path.join(output_dir, "ape_colormap.png")
    fig_ape.savefig(ape_plot_path, dpi=150, bbox_inches="tight")
    print(f"[evo] Saved APE colormap: {ape_plot_path}")

    plt.show()


def _prepare_output_dir(output_dir: str, save_frames: bool) -> str:
    """Create ``output_dir`` (and ``frames/`` if needed). Returns the frames dir path."""
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    return frames_dir


def _pose_error(current_pose: np.ndarray, target_pose: np.ndarray):
    """Returns (dist_m, angle_rad) between two camera-to-world poses, or (dist_m, None) if rotation fails."""
    dist_m = float(np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3]))
    try:
        curr_rot = R.from_matrix(current_pose[:3, :3])
        tgt_rot = R.from_matrix(target_pose[:3, :3])
        angle_rad = float(np.linalg.norm((tgt_rot.inv() * curr_rot).as_rotvec()))
    except Exception:
        angle_rad = None
    return dist_m, angle_rad


# -----------------------------------------------------------------------------
# Servoing loops (headless — no live plotting, async frame writes)
# -----------------------------------------------------------------------------

def visual_servoing_loop(
    scene: BaseScene,
    camera: Camera,
    target_image: np.ndarray,
    controller: BaseController,
    max_iterations: int = 200,
    dt: float = 0.1,
    error_tolerance: float = 5e-7,
    velocity_epsilon: float = 1e-9,
    target_pose: np.ndarray = None,
    output_dir: str = None,
    save_frames: bool = False,
    save_trajectory: bool = True,
    run_evo: bool = True,
    frame_format: str = "jpg",
    frame_quality: int = 90,
):
    """
    Executes the visual servoing control loop (headless).

    Args:
        scene, camera, target_image, controller: standard servoing inputs.
        max_iterations (int): Maximum number of control loop iterations.
        dt (float): Time step for velocity integration.
        error_tolerance (float): Stop when error norm falls below this.
        velocity_epsilon (float): Stop when velocity-gradient norm falls below this.
        target_pose (np.ndarray): Ground truth 4x4 target pose (Camera-to-World).
        output_dir (str): If set, trajectories (and optionally frames) go here.
        save_frames (bool): Save per-iteration [render|target] frames. OFF by default.
        save_trajectory (bool): Save estimated/ground-truth trajectories (TUM).
        run_evo (bool): After the loop, run evo to produce comparison plots.
        frame_format (str): "jpg" (default, fast) or "png" (lossless, slow).
        frame_quality (int): JPEG quality (1-100); ignored for PNG.
    """
    print("Starting Visual Servoing Loop...")

    frames_dir = None
    if output_dir is not None:
        frames_dir = _prepare_output_dir(output_dir, save_frames)

    frame_writer = None
    if output_dir is not None and save_frames:
        frame_writer = _AsyncFrameWriter(fmt=frame_format, quality=frame_quality)

    est_timestamps, est_poses = [], []
    gt_timestamps, gt_poses = [], []

    prev_v_c = None
    stop_reason = None
    final_i = 0
    target_resized = None  # cached on first iteration
    ext = "jpg" if frame_format.lower() in ("jpg", "jpeg") else "png"
    loop_start = time.time()

    interrupted = False
    try:
        for i in range(max_iterations):
            iter_start = time.time()

            # 1. Render current virtual view and depth map
            rendered_img = camera.render(scene)
            rendered_depth = camera.render_depth(scene)

            if target_resized is None:
                target_resized = _resize_target(target_image, rendered_img.shape[0], rendered_img.shape[1])

            # 2. Controller magic: compute velocity command based on the views
            v_c = controller.compute_velocity(
                rendered_img, rendered_depth, target_image, camera.K,
                current_pose=camera.pose, target_pose=target_pose
            )

            # 3. Apply control velocity to update camera pose
            camera.apply_velocity(v_c, dt=dt)

            # Cache pose once per iteration (property copies T_cw internally).
            current_pose = camera.pose

            # --- Trajectory logging (post-update pose) ---
            if output_dir is not None and save_trajectory:
                ts = i * dt
                est_timestamps.append(ts)
                est_poses.append(current_pose)
                if target_pose is not None:
                    gt_timestamps.append(ts)
                    gt_poses.append(np.asarray(target_pose, dtype=np.float64))

            # --- Per-iteration frame save (clean render|target, async) ---
            if frame_writer is not None:
                frame_path = os.path.join(frames_dir, f"frame_{i:05d}.{ext}")
                frame_writer.submit(frame_path, rendered_img, target_resized)

            v_norm = float(np.linalg.norm(v_c))
            e_norm = float(getattr(controller, 'current_error_norm', 0.0))
            final_i = i

            # Check early stopping conditions
            if e_norm > 0 and e_norm < error_tolerance:
                stop_reason = f"Error norm ({e_norm:.9f}) below tolerance ({error_tolerance})"
            elif prev_v_c is not None:
                v_c_grad = float(np.linalg.norm(v_c - prev_v_c) / dt)
                if v_c_grad < velocity_epsilon:
                    stop_reason = f"Velocity gradient ({v_c_grad:.9f}) below epsilon ({velocity_epsilon})"

            prev_v_c = v_c.copy()

            fps = 1.0 / (time.time() - iter_start + 1e-5)
            print(
                f"\rIter {i+1}/{max_iterations} | FPS {fps:5.1f} "
                f"| ||v_c|| {v_norm:.6e} | ||e|| {e_norm:.6e}",
                end="", flush=True,
            )

            if stop_reason:
                break
    except KeyboardInterrupt:
        interrupted = True
        print()  # newline after \r status line
        print("[interrupted] Ctrl+C received — persisting partial results before exiting...")
    finally:
        if frame_writer is not None:
            frame_writer.close()
        # Persist whatever was collected — guarantees partial runs leave artifacts.
        if output_dir is not None and save_trajectory and est_poses:
            est_path = os.path.join(output_dir, "trajectory_estimated.txt")
            _save_tum_trajectory(est_path, est_timestamps, est_poses)
            print(f"Saved estimated trajectory: {est_path} ({len(est_poses)} poses)")
            if gt_poses:
                gt_path = os.path.join(output_dir, "trajectory_groundtruth.txt")
                _save_tum_trajectory(gt_path, gt_timestamps, gt_poses)
                print(f"Saved ground-truth trajectory: {gt_path} ({len(gt_poses)} poses)")

    if not interrupted:
        print()  # terminate the \r-updated status line
        if stop_reason:
            print(f"Target reached at iteration {final_i + 1}: {stop_reason}")
        else:
            print(f"Did not converge within {max_iterations} iterations.")

        if target_pose is not None:
            dist_m, angle_rad = _pose_error(camera.pose, target_pose)
            if angle_rad is not None:
                print(f"Final Distance: {dist_m:.9f} m | Final Angle: {angle_rad:.9f} rad")
            else:
                print(f"Final Distance: {dist_m:.9f} m")

    print(f"Visual Servoing Loop Finished ({time.time() - loop_start:.2f}s total).")

    # --- Post-run: evo (only on a clean finish) ---
    if (not interrupted and output_dir is not None and save_trajectory
            and run_evo and est_poses and gt_poses):
        gt_path = os.path.join(output_dir, "trajectory_groundtruth.txt")
        est_path = os.path.join(output_dir, "trajectory_estimated.txt")
        _run_evo_evaluation(gt_path, est_path, output_dir)
    elif not interrupted and est_poses and not gt_poses:
        print("No ground-truth target pose was provided; skipping evo evaluation.")


def trajectory_servoing_loop(
    scene: BaseScene,
    camera: Camera,
    trajectory: list,
    controller: BaseController,
    max_iterations_per_target: int = 200,
    dt: float = 0.1,
    error_tolerance: float = 5e-7,
    velocity_epsilon: float = 1e-9,
    output_dir: str = None,
    save_frames: bool = False,
    save_trajectory: bool = True,
    run_evo: bool = True,
    frame_format: str = "jpg",
    frame_quality: int = 90,
):
    """
    Executes the visual servoing control loop over a sequence of targets (headless).

    Args:
        scene, camera, trajectory, controller: standard servoing inputs.
        trajectory (list): [(target_image1, target_pose1), (target_image2, target_pose2), ...].
        max_iterations_per_target (int): Max iterations for each target.
        dt, error_tolerance, velocity_epsilon: same as ``visual_servoing_loop``.
        output_dir (str): If set, trajectories (and optionally frames) go here.
        save_frames (bool): Save per-iteration [render|target] frames. OFF by default.
        save_trajectory (bool): Save estimated/ground-truth trajectories (TUM).
        run_evo (bool): After the loop, run evo to produce comparison plots.
        frame_format (str): "jpg" (default, fast) or "png" (lossless, slow).
        frame_quality (int): JPEG quality (1-100); ignored for PNG.
    """
    print(f"Starting Trajectory Servoing Loop for {len(trajectory)} targets...")

    frames_dir = None
    if output_dir is not None:
        frames_dir = _prepare_output_dir(output_dir, save_frames)

    frame_writer = None
    if output_dir is not None and save_frames:
        frame_writer = _AsyncFrameWriter(fmt=frame_format, quality=frame_quality)

    est_timestamps, est_poses = [], []
    gt_timestamps, gt_poses = [], []

    global_iteration = 0
    ext = "jpg" if frame_format.lower() in ("jpg", "jpeg") else "png"
    loop_start = time.time()

    interrupted = False
    try:
        for target_idx, (target_image, target_pose) in enumerate(trajectory):
            print(f"\n--- Moving to Target {target_idx + 1}/{len(trajectory)} ---")
            if hasattr(controller, 'reset'):
                controller.reset()

            prev_v_c = None
            stop_reason = None
            final_i = 0
            target_resized = None  # cached on first iteration of this waypoint

            for i in range(max_iterations_per_target):
                iter_start = time.time()

                rendered_img = camera.render(scene)
                rendered_depth = camera.render_depth(scene)

                if target_resized is None:
                    target_resized = _resize_target(
                        target_image, rendered_img.shape[0], rendered_img.shape[1]
                    )

                v_c = controller.compute_velocity(
                    rendered_img, rendered_depth, target_image, camera.K,
                    current_pose=camera.pose, target_pose=target_pose
                )

                camera.apply_velocity(v_c, dt=dt)

                # Cache pose once per iteration.
                current_pose = camera.pose

                # --- Trajectory logging (post-update pose) ---
                if output_dir is not None and save_trajectory:
                    ts = global_iteration * dt
                    est_timestamps.append(ts)
                    est_poses.append(current_pose)
                    if target_pose is not None:
                        gt_timestamps.append(ts)
                        gt_poses.append(np.asarray(target_pose, dtype=np.float64))

                # --- Per-iteration frame save (async, cached target resize) ---
                if frame_writer is not None:
                    frame_path = os.path.join(frames_dir, f"frame_{global_iteration:05d}.{ext}")
                    frame_writer.submit(frame_path, rendered_img, target_resized)

                v_norm = float(np.linalg.norm(v_c))
                e_norm = float(getattr(controller, 'current_error_norm', 0.0))

                global_iteration += 1
                final_i = i

                if e_norm > 0 and e_norm < error_tolerance:
                    stop_reason = f"Error norm ({e_norm:.9f}) below tolerance ({error_tolerance})"
                elif prev_v_c is not None:
                    v_c_grad = float(np.linalg.norm(v_c - prev_v_c) / dt)
                    if v_c_grad < velocity_epsilon:
                        stop_reason = f"Velocity gradient ({v_c_grad:.9f}) below epsilon ({velocity_epsilon})"

                prev_v_c = v_c.copy()

                fps = 1.0 / (time.time() - iter_start + 1e-5)
                print(
                    f"\rTarget {target_idx+1}/{len(trajectory)} | Iter {i+1}/{max_iterations_per_target} "
                    f"| FPS {fps:5.1f} | ||v_c|| {v_norm:.6e} | ||e|| {e_norm:.6e}",
                    end="", flush=True,
                )

                if stop_reason:
                    break

            print()  # terminate the \r-updated status line
            if stop_reason:
                print(f"Target {target_idx+1} reached at iteration {final_i + 1}: {stop_reason}")
            else:
                print(f"Target {target_idx+1} did not converge within {max_iterations_per_target} iterations.")

            if target_pose is not None:
                dist_m, angle_rad = _pose_error(camera.pose, target_pose)
                if angle_rad is not None:
                    print(f"  Final Distance: {dist_m:.9f} m | Final Angle: {angle_rad:.9f} rad")
                else:
                    print(f"  Final Distance: {dist_m:.9f} m")
    except KeyboardInterrupt:
        interrupted = True
        print()  # newline after \r status line
        print("[interrupted] Ctrl+C received — persisting partial results before exiting...")
    finally:
        if frame_writer is not None:
            frame_writer.close()
        # Persist whatever was collected so a Ctrl+C still leaves usable artifacts.
        if output_dir is not None and save_trajectory and est_poses:
            est_path = os.path.join(output_dir, "trajectory_estimated.txt")
            _save_tum_trajectory(est_path, est_timestamps, est_poses)
            print(f"Saved estimated trajectory: {est_path} ({len(est_poses)} poses)")
            if gt_poses:
                gt_path = os.path.join(output_dir, "trajectory_groundtruth.txt")
                _save_tum_trajectory(gt_path, gt_timestamps, gt_poses)
                print(f"Saved ground-truth trajectory: {gt_path} ({len(gt_poses)} poses)")

    print(f"\nTrajectory Tracking Finished ({time.time() - loop_start:.2f}s total).")

    # --- Post-run: evo (only on a clean finish) ---
    if (not interrupted and output_dir is not None and save_trajectory
            and run_evo and est_poses and gt_poses):
        gt_path = os.path.join(output_dir, "trajectory_groundtruth.txt")
        est_path = os.path.join(output_dir, "trajectory_estimated.txt")
        _run_evo_evaluation(gt_path, est_path, output_dir)
    elif not interrupted and est_poses and not gt_poses:
        print("No ground-truth target poses were provided; skipping evo evaluation.")
