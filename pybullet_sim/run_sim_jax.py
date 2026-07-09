import os
import sys
import time
import numpy as np
import torch
import cv2
import csv
from tqdm import tqdm
import pybullet as p
import importlib.util
import math
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.model.net_fast import ODANetFast
from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.control.jax_pa_mppi import JaxPAMPPIController
from pybullet_sim.env_generator import spawn_straight_racing_track
from pybullet_sim.export_3d_map import export_3d_html_map
from pybullet_sim.export_threejs import export_threejs_map

import jax
import jax.numpy as jnp

def get_camera_image(pos, rpy, client_id=0):
    # Simulate a gimbal that keeps pitch and roll at 0, only tracks yaw
    yaw = rpy[2]
    rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
    rot_mat = rot.as_matrix()
    
    forward = rot_mat[:, 1]
    up = rot_mat[:, 2]
    
    # Place camera slightly in front of the drone
    eye_pos = pos + forward * 0.15
    target = eye_pos + forward * 1.0
    
    view_matrix = p.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=target, cameraUpVector=up, physicsClientId=client_id)
    # PyBullet FOV is vertical FOV. 69 deg V-FOV, Aspect Ratio 1.0
    projection_matrix = p.computeProjectionMatrixFOV(69.0, 1.0, 0.1, 10.0, physicsClientId=client_id)
    width, height, rgb, depth, seg = p.getCameraImage(224, 224, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=client_id)
    rgb_np = np.reshape(rgb, (224, 224, 4)).astype(np.uint8)
    depth_buffer = np.reshape(depth, [224, 224])
    near, far = 0.1, 10.0
    linear_depth = far * near / (far - (far - near) * depth_buffer)
    return rgb_np, linear_depth

def get_third_person_image(pos, rpy, client_id=0, width=448, height=448):
    yaw = rpy[2]
    rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
    rot_mat = rot.as_matrix()
    forward = rot_mat[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    
    eye_pos = pos - forward * 1.5 + up * 0.5
    target_pos = pos + forward * 2.0
    view_matrix = p.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=target_pos, cameraUpVector=up, physicsClientId=client_id)
    projection_matrix = p.computeProjectionMatrixFOV(60.0, 1.0, 0.1, 30.0, physicsClientId=client_id)
    w_out, h_out, rgb, depth, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=client_id)
    rgb_np = np.reshape(rgb, (height, width, 4)).astype(np.uint8)
    return rgb_np

def main():
    import argparse
    parser = argparse.ArgumentParser(description="JAX UAV MPPI Obstacle Avoidance Simulation")
    parser.add_argument("--use-model", action="store_true", help="Use trained PyTorch model for depth estimation instead of ground truth")
    parser.add_argument("--onnx", type=str, default="", help="Path to ONNX model to use instead of PyTorch .pth")
    args = parser.parse_args()
    
    use_model_depth = args.use_model or (args.onnx != "")
    suffix = "_odanet" if use_model_depth else ""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.onnx:
        print(f"=> Mode: Using ONNX model ({args.onnx}) for depth map estimation.")
    elif use_model_depth:
        print("=> Mode: Using trained PyTorch model for depth map estimation.")
    else:
        print("=> Mode: Using ground truth depth map from simulator.")
    print("=> JAX PA-MPPI + Rolling Sparse Map active.")
    
    # 1. Init Gym Env (Bitcraze Crazyflie 2.X)
    INIT_XYZS = np.array([[0.0, 0.0, 1.0]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
    
    env = VelocityAviary(drone_model=DroneModel.CF2X,
                         num_drones=1,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         pyb_freq=240,
                         ctrl_freq=240,
                         gui=False,
                         record=False)
                         
    env.SPEED_LIMIT = 5.0
    env.target_yaw = 0.0
    
    import types
    
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
                        
        max_tilt_rad = np.radians(30.0)
        max_horizontal = self.GRAVITY * np.tan(max_tilt_rad)
        target_thrust[0] = np.clip(target_thrust[0], -max_horizontal, max_horizontal)
        target_thrust[1] = np.clip(target_thrust[1], -max_horizontal, max_horizontal)
        target_thrust[2] = max(0.1 * self.GRAVITY, target_thrust[2])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
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
    
    egl_spec = importlib.util.find_spec('eglRenderer')
    if egl_spec and egl_spec.origin:
        p.loadPlugin(egl_spec.origin, "_eglRendererPlugin", physicsClientId=PYB_CLIENT)
        print("=> EGL GPU Render Active")
        
    # 2. Init Model and load weights (if requested)
    model = None
    ort_session = None
    
    if args.onnx:
        import onnxruntime as ort
        print(f"Initializing ONNX Runtime session for {args.onnx}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(args.onnx, providers=providers)
    elif use_model_depth:
        model = ODANetFast(backbone_name='fastvit_t12').to(device)
        ckpt_p4 = "checkpoint/phase4/best_model_fast.pth"
        if os.path.exists(ckpt_p4):
            print(f"Loading Phase 4 Fast PyTorch weights from {ckpt_p4}...")
            ckpt = torch.load(ckpt_p4, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f"Error: Phase 4 weights not found at {ckpt_p4}! Please run Phase 4 training first.")
            sys.exit(1)
        model.eval()
    
    # 3. Init JAX Controllers & Mappers
    # Calculate fx, fy for PyBullet 69 deg vertical FOV on 224x224
    # f = (h/2) / tan(FOV/2) => 112 / tan(69/2 deg) = 112 / 0.6873 = 162.95
    fx = fy = 163.0
    cx = cy = 112.0
    sparse_map = RollingSparseMap(h=224, w=224, fx=fx, fy=fy, cx=cx, cy=cy, 
                                  max_frames=100, points_per_frame=2048, 
                                  voxel_size=0.15, max_unique=10000)
    controller = JaxPAMPPIController(horizon=30, num_samples=10000, dt=0.05)
    # Override vertical limits specifically for simulation to avoid spheres in 3D
    controller.u_min = controller.u_min.at[2].set(-1.0)
    controller.u_max = controller.u_max.at[2].set(1.0)
    controller.noise_sigma = controller.noise_sigma.at[2].set(0.5)  # Increased from 0.3: explore aggressive vertical maneuvers
    
    obs_dict, info = env.reset()
    
    # Spawn Straight Racing Track
    trees_data = spawn_straight_racing_track(client_id=PYB_CLIENT)
    
    # Virtual Track Boundaries (Static generation)
    track_width = 12.0
    virtual_walls = []
    for t_val in np.linspace(0.0, 1.2, 60):
        cx_t = 0.0
        cy_t = 40.0 * t_val
        angle_t = np.pi / 2.0
        wx_l = cx_t - (track_width / 2.0) * (-np.sin(angle_t))
        wy_l = cy_t - (track_width / 2.0) * (np.cos(angle_t))
        wx_r = cx_t + (track_width / 2.0) * (-np.sin(angle_t))
        wy_r = cy_t + (track_width / 2.0) * (np.cos(angle_t))
        for wz in [0.5, 1.0, 1.5]:
            virtual_walls.extend([[wx_l, wy_l, wz], [wx_r, wy_r, wz]])
            
    static_virtual_walls_jax = jnp.array(virtual_walls, dtype=jnp.float32)
    
    # Video
    os.makedirs("pybullet_sim", exist_ok=True)
    out_video_path = f"pybullet_sim/flight_gym_jax{suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_video_path, fourcc, 20.0, (672, 448))
    
    drone_path = []
    telemetry_data = []
    mppi_history = []
    goal_reached = False
    goal_y = 40.0
    goal_threshold = 1.5  # meters from goal Y to consider "reached"
    
    MAX_STEPS = 500
    pbar = tqdm(range(MAX_STEPS), desc="JAX Flight")
    for step in pbar:
        state = obs_dict[0]
        pos = state[0:3]
        quat = state[3:7]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        
        drone_path.append(pos.copy())
        
        # Render cameras using Gimbal-stabilized RPY
        rgb_fpv, depth_gt = get_camera_image(pos, rpy, PYB_CLIENT)
        
        # Perception
        if use_model_depth:
            rgb_input = rgb_fpv[:, :, :3].astype(np.float32) / 255.0
            
            if ort_session is not None:
                # ONNX Inference
                rgb_tensor_np = np.expand_dims(np.transpose(rgb_input, (2, 0, 1)), axis=0)
                ort_inputs = {ort_session.get_inputs()[0].name: rgb_tensor_np}
                ort_outs = ort_session.run(None, ort_inputs)
                # Output order from export_onnx.py: ['depth', 'confidence', 'distance']
                depth_tensor = torch.from_numpy(ort_outs[0]).to(device)
                conf_tensor = torch.from_numpy(ort_outs[1]).to(device)
            else:
                # PyTorch Inference
                rgb_tensor = torch.from_numpy(rgb_input).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(rgb_tensor)
                    depth_tensor = outputs["depth"]
                    
            depth_np_for_vis = depth_tensor[0, 0].cpu().numpy()
        else:
            depth_tensor = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).to(device)
            depth_np_for_vis = depth_gt
        
        # JAX Sparse Mapping
        # Use yaw-only rotation matrix for projection because camera is gimbal-stabilized
        rot_yaw_only = Rotation.from_euler('xyz', [0.0, 0.0, rpy[2]]).as_matrix()
        R_w_jax = jnp.array(rot_yaw_only, dtype=jnp.float32)
        p_w_jax = jnp.array(pos, dtype=jnp.float32)
        
        unique_points_jax = sparse_map.update(
            depth_tensor, 
            R_w_jax, p_w_jax, pitch_deg=0.0, # Gimbal removes pitch
            max_depth=6.0
        )
        
        # Filter floor points (ignore points too close to ground z <= 0.15)
        unique_points_jax = jnp.where(
            unique_points_jax[:, 2:3] <= 0.15,
            1e5,
            unique_points_jax
        )
        obs_points_jax = unique_points_jax
        
        # Combine perceived obstacles with static virtual track boundaries
        all_obs_jax = jnp.vstack([obs_points_jax, static_virtual_walls_jax])
        
        # MPPI Planning
        t_curr = np.clip(pos[1] / 40.0, 0.0, 1.0)
        t_target = min(1.0, t_curr + 0.15) 
        wp_x = 0.0
        wp_y = 40.0 * t_target
        goal_pos_jax = jnp.array([wp_x, wp_y, 1.0], dtype=jnp.float32)
        state_0_jax = jnp.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], rpy[2]], dtype=jnp.float32)
        
        action, best_traj, all_trajs, costs = controller.compute_action(state_0_jax, goal_pos_jax, all_obs_jax)
        
        # Save history for visualization (Subsample to prevent memory/CPU explosion)
        if step % 5 == 0:
            # 1. Pull costs to CPU (small array: 10000 floats)
            costs_np = np.array(costs)
            # 2. Get top 30 indices
            top_indices = np.argsort(costs_np)[:30]
            # 3. Slice the trajectories ON THE GPU, then pull to CPU (only 30 paths!)
            top_cands = np.array(all_trajs[top_indices, :, :3])
            
            # Subsample points
            unique_pts_np = np.array(unique_points_jax)
            valid_pts = unique_pts_np[unique_pts_np[:, 2] < 1e4]
            if len(valid_pts) > 1000:
                sub_idx = np.random.choice(len(valid_pts), 1000, replace=False)
                html_pts = valid_pts[sub_idx]
            else:
                html_pts = valid_pts
                
            mppi_history.append({
                'frame_idx': step,
                'p_w': pos.copy(),
                'yaw': float(rpy[2]),
                'known_points': html_pts.tolist(),
                'best_path': np.array(best_traj)[:, :3],
                'candidate_paths': top_cands
            })
        
        # Step Gym Env
        gym_action = np.zeros((1, 4))
        gym_action[0, :] = np.array(action)
            
        for _ in range(12):
            obs_dict, reward, terminated, truncated, info = env.step(gym_action)
            
        # Visualization Dashboard
        rgb_tpv = get_third_person_image(pos, rpy, PYB_CLIENT, width=448, height=448)
        bgr_fpv = cv2.cvtColor(rgb_fpv, cv2.COLOR_RGBA2BGR)
        bgr_tpv = cv2.cvtColor(rgb_tpv, cv2.COLOR_RGBA2BGR)
        
        depth_norm = np.clip((depth_np_for_vis - 0.1) / (6.0 - 0.1), 0.0, 1.0)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        bgr_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
        
        cv2.putText(bgr_tpv, "3rd Person View (TPV)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bgr_fpv, "FPV Camera (RGB)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        depth_label = "FPV Depth Map (ODANetFast)" if use_model_depth else "FPV Depth Map (Ground Truth)"
        cv2.putText(bgr_depth, depth_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        right_column = np.vstack((bgr_fpv, bgr_depth))
        combined_frame = np.hstack((bgr_tpv, right_column))
        out_vid.write(combined_frame)
        
        valid_pts_count = int(jnp.sum(obs_points_jax[:, 2] < 1e4))
        
        # Compute nearest obstacle distance from trees_data
        nearest_obs_dist = float('inf')
        for t in trees_data:
            if t['type'] == 'cylinder':
                # 2D distance to cylinder axis minus radius
                d_horiz = np.sqrt((pos[0] - t['x'])**2 + (pos[1] - t['y'])**2) - t['radius']
                nearest_obs_dist = min(nearest_obs_dist, d_horiz)
            elif t['type'] == 'sphere':
                # 3D distance to sphere center minus radius
                d_3d = np.sqrt((pos[0] - t['x'])**2 + (pos[1] - t['y'])**2 + (pos[2] - t['z'])**2) - t['radius']
                nearest_obs_dist = min(nearest_obs_dist, d_3d)
        nearest_obs_dist = max(0.0, nearest_obs_dist)  # clamp negative = collision
        
        telemetry_data.append({
            'step': step,
            'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'roll': rpy[0], 'pitch': rpy[1], 'yaw': rpy[2],
            'action_vx': gym_action[0, 0], 'action_vy': gym_action[0, 1], 'action_vz': gym_action[0, 2],
            'action_yaw_rate': gym_action[0, 3],
            'nearest_obs_dist': nearest_obs_dist
        })
        
        pbar.set_postfix({"Pts": valid_pts_count, "ObsDist": f"{nearest_obs_dist:.2f}"})
        
        # Early termination: goal reached
        if pos[1] >= goal_y - goal_threshold:
            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            if speed < 1.0 or pos[1] >= goal_y:
                goal_reached = True
                print(f"\n==> GOAL REACHED at step {step}! pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), speed={speed:.2f} m/s")
                break
        
    status = "GOAL REACHED" if goal_reached else f"TIMEOUT ({MAX_STEPS} steps)"
    print(f"Simulation finished. Status: {status}, Steps: {len(telemetry_data)}")
    out_vid.release()
    export_3d_html_map(trees_data, drone_path, np.array([0.0, 40.0, 1.0]), f"pybullet_sim/flight_gym_jax_map{suffix}.html")
    export_threejs_map(mppi_history, drone_path, np.array([0.0, 40.0, 1.0]), trees_data, output_dir=f"pybullet_sim/viewer{suffix}")
    
    csv_path = f"pybullet_sim/telemetry_jax{suffix}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['step', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'roll', 'pitch', 'yaw', 'action_vx', 'action_vy', 'action_vz', 'action_yaw_rate', 'nearest_obs_dist']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry_data)
    
    # Print summary statistics
    import statistics
    obs_dists = [row['nearest_obs_dist'] for row in telemetry_data]
    min_clearance = min(obs_dists)
    collisions = sum(1 for d in obs_dists if d < 0.05)
    print(f"Telemetry saved to {csv_path}")
    print(f"=== Flight Summary ===")
    print(f"  Steps flown: {len(telemetry_data)}")
    print(f"  Min clearance: {min_clearance:.3f}m")
    print(f"  Near-collision events (d < 0.05m): {collisions}")
    print(f"  Mean clearance: {statistics.mean(obs_dists):.3f}m")
    
    env.close()

if __name__ == "__main__":
    main()
