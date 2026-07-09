"""
Batch Simulation Benchmark for PA-MPPI UAV Obstacle Avoidance.

Runs N random trials with different obstacle layouts (random seeds),
collecting per-trial metrics for statistical analysis.

Usage:
    python pybullet_sim/batch_sim_jax.py --num-trials 500
    python pybullet_sim/batch_sim_jax.py --num-trials 50 --max-steps 300  # quick test
"""

import os
import sys
import time
import io
import argparse
import csv
import random
import statistics
import numpy as np
import torch
import pybullet as p
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import traceback
from PIL import Image
from transformers import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.control.jax_pa_mppi import JaxPAMPPIController
from pybullet_sim.env_generator import spawn_straight_racing_track
from src.model.net_fast import ODANetFast

import jax
import jax.numpy as jnp


# Context manager to suppress C-level stdout/stderr (EGL, PyBullet prints)
from contextlib import contextmanager

@contextmanager
def suppress_c_output():
    """Suppress C-level stdout/stderr that bypass Python's sys.stdout."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    # Also save Python-level
    saved_py_stdout = sys.stdout
    saved_py_stderr = sys.stderr
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        sys.stdout = saved_py_stdout
        sys.stderr = saved_py_stderr


# ============================================================================
# Randomized Track Generator (Seeded)
# ============================================================================
def spawn_random_track(seed, client_id=0):
    """
    Spawn a random obstacle track with the given seed.
    Returns list of obstacle dicts.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    trees = []
    
    track_width = 7.0
    wall_thickness = 0.2
    wall_height = 2.5
    
    # Track boundary walls (same geometry every trial)
    num_segments = 25
    t_vals = np.linspace(0, 1.0, num_segments + 1)
    for i in range(num_segments):
        y1 = 40.0 * t_vals[i]
        y2 = 40.0 * t_vals[i+1]
        cx, cy = 0.0, (y1 + y2) / 2.0
        length = y2 - y1
        angle = np.pi / 2.0
        
        for side in [-1, 1]:
            wx = cx + side * (track_width / 2.0) * (-np.sin(angle))
            wy = cy + side * (track_width / 2.0) * (np.cos(angle))
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height], physicsClientId=client_id)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height], rgbaColor=[0.15, 0.15, 0.15, 1.0], physicsClientId=client_id)
            orn = p.getQuaternionFromEuler([0, 0, angle])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                              basePosition=[wx, wy, wall_height], baseOrientation=orn, physicsClientId=client_id)
    
    # Randomize number of obstacles
    num_cylinders = rng.randint(6, 12)
    num_sphere_rows = rng.randint(4, 7)
    
    # Cylinders in first half (y: 2.0 → 18.0)
    y_step = (18.0 - 2.0) / num_cylinders
    for i in range(num_cylinders):
        oy = rng.uniform(2.0 + i * y_step, 2.0 + (i + 1) * y_step)
        ox = rng.uniform(-track_width/2 + 0.8, track_width/2 - 0.8)
        radius = rng.uniform(0.15, 0.40)
        height = 5.0
        
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.9, 0.3, 0.1, 1], physicsClientId=client_id)
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                    basePosition=[ox, oy, height/2], physicsClientId=client_id)
        trees.append({'id': body_id, 'x': ox, 'y': oy, 'z': 0.0, 'radius': radius, 'height': height, 'type': 'cylinder'})
    
    # Floating spheres in second half (y: 20.0 → 38.0)
    sphere_y_start = 20.0
    sphere_y_end = 38.0
    y_gap = (sphere_y_end - sphere_y_start) / num_sphere_rows
    
    for row_i in range(num_sphere_rows):
        oy = rng.uniform(sphere_y_start + row_i * y_gap, sphere_y_start + (row_i + 1) * y_gap)
        num_in_row = rng.randint(2, 4)
        
        heights = [rng.uniform(0.35, 0.75), rng.uniform(0.9, 1.25), rng.uniform(1.4, 1.8)]
        rng.shuffle(heights)
        
        x_positions = sorted([rng.uniform(-track_width/2 + 0.9, track_width/2 - 0.9) for _ in range(num_in_row)])
        
        for j, ox in enumerate(x_positions):
            z = heights[j % len(heights)]
            radius = rng.uniform(0.20, 0.45)
            
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.1, 0.6, 0.9, 1.0], physicsClientId=client_id)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                        basePosition=[ox, oy, z], physicsClientId=client_id)
            trees.append({'id': body_id, 'x': ox, 'y': oy, 'z': z, 'radius': radius, 'height': radius*2, 'type': 'sphere'})
    
    return trees


# ============================================================================
# Camera Helper
# ============================================================================
def get_camera_image(pos, rpy, client_id=0):
    yaw = rpy[2]
    rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
    rot_mat = rot.as_matrix()
    forward = rot_mat[:, 1]
    up = rot_mat[:, 2]
    eye_pos = pos + forward * 0.15
    target = eye_pos + forward * 1.0
    view_matrix = p.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=target, cameraUpVector=up, physicsClientId=client_id)
    projection_matrix = p.computeProjectionMatrixFOV(69.0, 1.0, 0.1, 10.0, physicsClientId=client_id)
    width, height, rgb, depth, seg = p.getCameraImage(224, 224, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=client_id)
    rgb_np = np.reshape(rgb, (224, 224, 4)).astype(np.uint8)
    depth_buffer = np.reshape(depth, [224, 224])
    near, far = 0.1, 10.0
    linear_depth = far * near / (far - (far - near) * depth_buffer)
    return rgb_np, linear_depth


# ============================================================================
# Nearest Obstacle Distance Calculator
# ============================================================================
def compute_nearest_obs_dist(pos, trees_data):
    nearest = float('inf')
    for t in trees_data:
        if t['type'] == 'cylinder':
            d = np.sqrt((pos[0] - t['x'])**2 + (pos[1] - t['y'])**2) - t['radius']
        elif t['type'] == 'sphere':
            d = np.sqrt((pos[0] - t['x'])**2 + (pos[1] - t['y'])**2 + (pos[2] - t['z'])**2) - t['radius']
        else:
            continue
        nearest = min(nearest, d)
    return max(0.0, nearest)


# ============================================================================
# Single Trial Runner
# ============================================================================
def run_single_trial(seed, controller, sparse_map, device, max_steps=500, model=None, model_type=None):
    """
    Run one complete trial with the given seed.
    Returns a dict of metrics for this trial.
    """
    import math
    import types
    import importlib.util
    
    INIT_XYZS = np.array([[0.0, 0.0, 1.0]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
    
    # Suppress PyBullet's verbose [INFO] output during env creation
    with suppress_c_output():
        env = VelocityAviary(drone_model=DroneModel.CF2X, num_drones=1,
                             initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS,
                             physics=Physics.PYB, pyb_freq=240, ctrl_freq=240,
                             gui=False, record=False)
    env.SPEED_LIMIT = 5.0
    env.target_yaw = 0.0
    
    # Custom PID with thrust clamping (exact copy from run_sim_jax.py)
    def custom_dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
        # Thrust clamping (critical for stability!)
        max_tilt_rad = np.radians(30.0)
        max_horizontal = self.GRAVITY * np.tan(max_tilt_rad)
        target_thrust[0] = np.clip(target_thrust[0], -max_horizontal, max_horizontal)
        target_thrust[1] = np.clip(target_thrust[1], -max_horizontal, max_horizontal)
        target_thrust[2] = max(0.1 * self.GRAVITY, target_thrust[2])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        return thrust, target_euler, pos_e

    for k in range(env.NUM_DRONES):
        env.ctrl[k]._dslPIDPositionControl = types.MethodType(custom_dslPIDPositionControl, env.ctrl[k])

    # Velocity smoothing preprocessor (critical - prevents erratic commands!)
    def custom_preprocess_action(self, action):
        if not hasattr(self, 'smoothed_target_v'):
            self.smoothed_target_v = np.zeros((self.NUM_DRONES, 3))
            self.smoothed_yaw_rate = np.zeros(self.NUM_DRONES)
            
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            raw_target_v = action[k, 0:3]
            raw_yaw_rate = action[k, 3]
            
            max_accel = 5.0
            max_dv = max_accel * self.CTRL_TIMESTEP
            
            dv = raw_target_v - self.smoothed_target_v[k]
            dv = np.clip(dv, -max_dv, max_dv)
            self.smoothed_target_v[k] += dv
            
            dyaw = raw_yaw_rate - self.smoothed_yaw_rate[k]
            dyaw = np.clip(dyaw, -max_dv, max_dv)
            self.smoothed_yaw_rate[k] += dyaw
            
            target_v = self.smoothed_target_v[k]
            yaw_rate = self.smoothed_yaw_rate[k]
            
            target_pos = state[0:3] + target_v * 0.1
            
            current_yaw = state[9]
            self.target_yaw = current_yaw + yaw_rate * self.CTRL_TIMESTEP
            
            temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=target_pos,
                                                    target_rpy=np.array([0.0, 0.0, self.target_yaw]),
                                                    target_vel=target_v,
                                                    target_rpy_rates=np.array([0.0, 0.0, yaw_rate]))
            rpm[k, :] = temp
        return rpm
        
    env._preprocessAction = types.MethodType(custom_preprocess_action, env)
    
    PYB_CLIENT = env.getPyBulletClient()
    
    # Load EGL plugin for hardware OpenGL (required for ER_BULLET_HARDWARE_OPENGL)
    egl_spec = importlib.util.find_spec('eglRenderer')
    if egl_spec and egl_spec.origin:
        with suppress_c_output():
            p.loadPlugin(egl_spec.origin, "_eglRendererPlugin", physicsClientId=PYB_CLIENT)
    
    obs_dict, info = env.reset()
    
    # Spawn track using the same proven generator as run_sim_jax.py
    # but with different random seed for obstacle placement variety
    random.seed(seed)
    with suppress_c_output():
        trees_data = spawn_straight_racing_track(client_id=PYB_CLIENT, seed=seed)
    num_obstacles = len(trees_data)
    
    # Virtual walls
    track_width = 12.0
    virtual_walls = []
    for t_val in np.linspace(0.0, 1.2, 60):
        cx_t = 0.0
        cy_t = 40.0 * t_val
        for side in [-1, 1]:
            wx = side * (track_width / 2.0)
            for wz in [0.5, 1.0, 1.5]:
                virtual_walls.append([wx, cy_t, wz])
    static_virtual_walls_jax = jnp.array(virtual_walls, dtype=jnp.float32)
    
    # Reset sparse map buffer for this trial
    sparse_map.buffer = jnp.ones_like(sparse_map.buffer) * 1e5
    sparse_map.frame_idx = 0
    
    # Reset controller warm-start
    controller.U_guess = jnp.zeros((controller.K, 4))
    controller.rng_key = jax.random.PRNGKey(seed)
    
    # ── Run Trial ──
    goal_y = 40.0
    goal_reached = False
    collision_detected = False
    
    # Metrics accumulators
    all_obs_dists = []
    all_speeds = []
    all_positions = []
    all_actions = []
    all_maes = []
    all_rmses = []
    min_clearance = float('inf')
    
    t_start = time.perf_counter()
    
    for step in range(max_steps):
        state = obs_dict[0]
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        
        all_positions.append(pos.copy())
        speed = np.linalg.norm(vel)
        all_speeds.append(speed)
        
        # Nearest obstacle distance
        obs_dist = compute_nearest_obs_dist(pos, trees_data)
        all_obs_dists.append(obs_dist)
        min_clearance = min(min_clearance, obs_dist)
        
        # Collision check (clearance < 0.02m or drone crashed)
        if obs_dist < 0.02:
            collision_detected = True
        if pos[2] < 0.08:  # crashed into ground
            collision_detected = True
            break
        
        # Depth from simulator (GT)
        rgb_np, depth_gt = get_camera_image(pos, rpy, PYB_CLIENT)
        
        if model is not None:
            rgb_input = rgb_np[:, :, :3]
            
            if model_type == 'da2':
                pil_image = Image.fromarray(rgb_input)
                hf_res = model(pil_image)
                depth_hf_raw = np.array(hf_res["depth"]).astype(np.float32)
                # Convert HF relative depth (0-255, 255=close) to rough metric depth (0-10, 0=close)
                depth_pred = 10.0 * (1.0 - depth_hf_raw / 255.0)
            else:
                # ODANetFast Inference
                img_tensor = torch.from_numpy(rgb_input.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    depth_pred = model(img_tensor)["depth"][0, 0].cpu().numpy()
            
            # Compute Perception Errors
            err = np.abs(depth_pred - depth_gt)
            all_maes.append(float(np.mean(err)))
            all_rmses.append(float(np.sqrt(np.mean(err**2))))
            
            depth_input = depth_pred
        else:
            depth_input = depth_gt
            
        depth_tensor = torch.from_numpy(depth_input).unsqueeze(0).unsqueeze(0).to(device)
        
        # Sparse mapping
        rot_yaw_only = Rotation.from_euler('xyz', [0.0, 0.0, rpy[2]]).as_matrix()
        R_w_jax = jnp.array(rot_yaw_only, dtype=jnp.float32)
        p_w_jax = jnp.array(pos, dtype=jnp.float32)
        unique_points_jax = sparse_map.update(depth_tensor, R_w_jax, p_w_jax,
                                               pitch_deg=0.0, max_depth=6.0)
        
        # Filter floor points
        unique_points_jax = jnp.where(unique_points_jax[:, 2:3] <= 0.15, 1e5, unique_points_jax)
        all_obs_jax = jnp.vstack([unique_points_jax, static_virtual_walls_jax])
        
        # MPPI planning
        t_curr = np.clip(pos[1] / 40.0, 0.0, 1.0)
        t_target = min(1.0, t_curr + 0.15)
        goal_pos_jax = jnp.array([0.0, 40.0 * t_target, 1.0], dtype=jnp.float32)
        state_0_jax = jnp.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], rpy[2]], dtype=jnp.float32)
        
        action, best_traj, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, all_obs_jax)
        all_actions.append(np.array(action))
        
        # Step env
        gym_action = np.zeros((1, 4))
        gym_action[0, :] = np.array(action)
        for _ in range(12):
            obs_dict, reward, terminated, truncated, info = env.step(gym_action)
        
        # Goal check
        if pos[1] >= goal_y - 1.5:
            if speed < 1.0 or pos[1] >= goal_y:
                goal_reached = True
                break
    
    t_elapsed = time.perf_counter() - t_start
    total_steps = len(all_positions)
    
    with suppress_c_output():
        env.close()
    
    # ── Compute Metrics ──
    positions = np.array(all_positions)
    actions = np.array(all_actions) if all_actions else np.zeros((1, 4))
    
    # 1. Success
    success = 1 if goal_reached and not collision_detected else 0
    
    # 2. TCR (Trajectory Collision Rate): fraction of steps where clearance < safe_radius)
    collision_threshold = 0.15  # within obstacle's influence zone
    tcr = sum(1 for d in all_obs_dists if d < collision_threshold) / max(1, total_steps)
    
    # 3. Min Clearance (d_min)
    d_min = min_clearance
    
    # 4. Mean Clearance
    d_mean = statistics.mean(all_obs_dists) if all_obs_dists else 0.0
    
    # 5. Trajectory Smoothness (Jerk = 2nd derivative of positions)
    if len(positions) >= 3:
        jerk = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        jerk_norm = np.linalg.norm(jerk, axis=-1)
        mean_jerk = float(np.mean(jerk_norm))
    else:
        mean_jerk = 0.0
    
    # 6. Goal Progress Rate (how much of the 40m was covered per step)
    final_y = positions[-1, 1] if len(positions) > 0 else 0.0
    goal_progress = final_y / 40.0  # 0.0 to 1.0+
    progress_per_step = goal_progress / max(1, total_steps)
    
    # 7. Mean Speed
    mean_speed = statistics.mean(all_speeds) if all_speeds else 0.0
    
    # 8. Total distance traveled
    if len(positions) >= 2:
        diffs = np.diff(positions, axis=0)
        total_distance = float(np.sum(np.linalg.norm(diffs, axis=-1)))
    else:
        total_distance = 0.0
    
    # 9. Action smoothness (mean delta-u)
    if len(actions) >= 2:
        delta_u = np.diff(actions, axis=0)
        action_smoothness = float(np.mean(np.linalg.norm(delta_u, axis=-1)))
    else:
        action_smoothness = 0.0
    
    # 10. Time to complete (sim time = steps * 12 * (1/240) seconds)
    sim_time = total_steps * 12 / 240.0  # each step = 12 physics substeps at 240Hz
    
    # 11. Perception Errors
    mean_mae = statistics.mean(all_maes) if all_maes else 0.0
    mean_rmse = statistics.mean(all_rmses) if all_rmses else 0.0
    
    return {
        'seed': seed,
        'success': success,
        'collision': 1 if collision_detected else 0,
        'goal_reached': 1 if goal_reached else 0,
        'num_obstacles': num_obstacles,
        'total_steps': total_steps,
        'sim_time_s': round(sim_time, 3),
        'wall_time_s': round(t_elapsed, 2),
        'final_y': round(final_y, 3),
        'goal_progress': round(goal_progress, 4),
        'tcr': round(tcr, 5),
        'd_min': round(d_min, 4),
        'd_mean': round(d_mean, 4),
        'mean_jerk': round(mean_jerk, 6),
        'action_smoothness': round(action_smoothness, 5),
        'mean_speed': round(mean_speed, 4),
        'total_distance': round(total_distance, 3),
        'progress_per_step': round(progress_per_step, 6),
        'mean_mae': round(mean_mae, 4),
        'mean_rmse': round(mean_rmse, 4),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Batch benchmark for PA-MPPI UAV")
    parser.add_argument('--num-trials', type=int, default=500, help='Number of random trials')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per trial')
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed')
    parser.add_argument('--output', type=str, default='pybullet_sim/batch_benchmark.csv', help='Output CSV path')
    parser.add_argument('--use-model', action='store_true', help='Use ODANetFast instead of GT depth')
    parser.add_argument('--use-da2', action='store_true', help='Use Depth Anything V2 Teacher Model instead of GT depth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== PA-MPPI Batch Benchmark ===")
    print(f"Trials: {args.num_trials}, Max steps: {args.max_steps}, Start seed: {args.start_seed}")
    print(f"Device: {device}")
    
    model = None
    model_type = None
    if args.use_da2:
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        print(f"Loading Depth Anything V2 on device: {device} with precision: {dtype}...")
        model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if device.type == 'cuda' else -1, dtype=dtype)
        model_type = 'da2'
        print("=> Mode: Depth Anything V2 Closed-Loop Perception")
    elif args.use_model:
        model = ODANetFast(backbone_name='fastvit_t12').to(device)
        ckpt_path = "checkpoint/phase4/best_model_fast.pth"
        if os.path.exists(ckpt_path):
            print(f"Loading ODANet weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("WARNING: Model weights not found!")
        model.eval()
        model_type = 'odanet'
        print("=> Mode: ODANet Closed-Loop Perception")
    else:
        print("=> Mode: Ground Truth Depth")
    
    # Initialize shared controller and mapper (once, reuse across trials)
    fx = fy = 163.0
    cx = cy = 112.0
    sparse_map = RollingSparseMap(h=224, w=224, fx=fx, fy=fy, cx=cx, cy=cy,
                                  max_frames=100, points_per_frame=2048,
                                  voxel_size=0.15, max_unique=10000)
    controller = JaxPAMPPIController(horizon=30, num_samples=10000, dt=0.05)
    controller.u_min = controller.u_min.at[2].set(-1.0)
    controller.u_max = controller.u_max.at[2].set(1.0)
    controller.noise_sigma = controller.noise_sigma.at[2].set(0.5)
    
    # Run trials
    results = []
    successes = 0
    collisions = 0
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    FIELDNAMES = ['seed', 'success', 'collision', 'goal_reached', 'num_obstacles',
                  'total_steps', 'sim_time_s', 'wall_time_s', 'final_y', 'goal_progress',
                  'tcr', 'd_min', 'd_mean', 'mean_jerk', 'action_smoothness',
                  'mean_speed', 'total_distance', 'progress_per_step', 'mean_mae', 'mean_rmse']
    
    # Open CSV and write header immediately
    csvfile = open(args.output, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()
    csvfile.flush()
    
    pbar = tqdm(range(args.num_trials), desc="Batch Benchmark")
    for trial_idx in pbar:
        seed = args.start_seed + trial_idx
        
        try:
            metrics = run_single_trial(seed, controller, sparse_map, device, args.max_steps, model, model_type)
            results.append(metrics)
            
            successes += metrics['success']
            collisions += metrics['collision']
            success_rate = successes / (trial_idx + 1) * 100
            
            pbar.set_postfix({
                'SR': f'{success_rate:.1f}%',
                'Col': collisions,
                'Seed': seed,
                'Steps': metrics['total_steps'],
            })
        except Exception as e:
            traceback.print_exc()
            print(f"\n[ERROR] Trial {trial_idx} (seed={seed}) failed: {e}")
            metrics = {
                'seed': seed, 'success': 0, 'collision': 1, 'goal_reached': 0,
                'num_obstacles': 0, 'total_steps': 0, 'sim_time_s': 0, 'wall_time_s': 0,
                'final_y': 0, 'goal_progress': 0, 'tcr': 1.0, 'd_min': 0, 'd_mean': 0,
                'mean_jerk': 0, 'action_smoothness': 0, 'mean_speed': 0, 'total_distance': 0,
                'progress_per_step': 0, 'mean_mae': 0, 'mean_rmse': 0,
            }
            results.append(metrics)
        
        # Write this trial's result to CSV immediately
        writer.writerow(metrics)
        csvfile.flush()
    
    csvfile.close()
    
    # Print summary
    n = len(results)
    success_list = [r['success'] for r in results]
    tcr_list = [r['tcr'] for r in results]
    dmin_list = [r['d_min'] for r in results]
    dmean_list = [r['d_mean'] for r in results]
    jerk_list = [r['mean_jerk'] for r in results]
    speed_list = [r['mean_speed'] for r in results]
    time_list = [r['sim_time_s'] for r in results if r['success'] == 1]
    progress_list = [r['goal_progress'] for r in results]
    
    print(f"\n{'='*60}")
    print(f"  PA-MPPI BATCH BENCHMARK RESULTS ({n} trials)")
    print(f"{'='*60}")
    print(f"  Success Rate:       {sum(success_list)/n*100:.1f}%  ({sum(success_list)}/{n})")
    print(f"  Collision Rate:     {sum(r['collision'] for r in results)/n*100:.1f}%")
    print(f"  Mean TCR:           {statistics.mean(tcr_list):.4f}")
    print(f"  Min Clearance:      mean={statistics.mean(dmin_list):.3f}m, min={min(dmin_list):.3f}m")
    print(f"  Mean Clearance:     {statistics.mean(dmean_list):.3f}m")
    print(f"  Mean Jerk:          {statistics.mean(jerk_list):.5f}")
    print(f"  Mean Speed:         {statistics.mean(speed_list):.3f} m/s")
    print(f"  Goal Progress:      mean={statistics.mean(progress_list):.3f}")
    
    mae_list = [r['mean_mae'] for r in results if r['mean_mae'] > 0]
    if mae_list:
        print(f"  Depth MAE:          {statistics.mean(mae_list):.3f} m")
        
    if time_list:
        print(f"  Sim Time (success): mean={statistics.mean(time_list):.2f}s, std={statistics.stdev(time_list) if len(time_list)>1 else 0:.2f}s")
    print(f"{'='*60}")
    print(f"  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
