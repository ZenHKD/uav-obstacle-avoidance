import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.net_fast import ODANetFast
from src.control.jax_pa_mppi import JaxPAMPPIController
from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.train.phase4.oda_utils import get_drone_pose, get_calib_rotation, calibrate_camera_pitch_from_first_frame

import jax
import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser(description="Benchmark TCR and d_min for Phase 4 JAX Pipeline")
    parser.add_argument('--checkpoint', type=str, default="checkpoint/phase4/best_model_fast.pth", help="Path to ODANetFast checkpoint")
    parser.add_argument('--video_path', type=str, default="dataset/11/11.avi", help="Path to input video")
    parser.add_argument('--overview_csv', type=str, default="dataset/trial_overview.csv", help="Path to trial_overview.csv")
    parser.add_argument('--pitch_deg', type=float, default=None, help="Fixed camera mount pitch. If None, uses auto-calibration.")
    parser.add_argument('--gimbal_mode', action='store_true', help="Discard pitch/roll from OptiTrack and simulate a Gimbal")
    parser.add_argument('--conf_thresh', type=float, default=0.75, help="Confidence threshold")
    parser.add_argument('--use_gt_depth', action='store_true', help="Use Ground Truth depth instead of network predictions")
    args = parser.parse_args()

    # Extract Sequence ID
    try:
        seq_id = int(os.path.basename(os.path.dirname(args.video_path)))
        print(f"Detected Sequence ID: {seq_id}")
    except ValueError:
        print(f"Warning: Could not extract sequence ID from {args.video_path}. Defaulting to 1.")
        seq_id = 1

    if not os.path.exists(args.overview_csv):
        print(f"Error: {args.overview_csv} not found!")
        return

    df = pd.read_csv(args.overview_csv)
    trial_rows = df[df['Sequence'] == seq_id]
    
    if len(trial_rows) == 0:
        print(f"Error: No ground truth data found for Sequence {seq_id}")
        return

    gt_obstacles = []
    for _, row in trial_rows.iterrows():
        obs_x = float(row['Obstacle x'])
        obs_y = -float(row['Obstacle z'])
        obs_z = float(row['Obstacle y'])
        gt_obstacles.append(np.array([obs_x, obs_y, obs_z], dtype=np.float32))
        
    print(f"Loaded {len(gt_obstacles)} GT Obstacles")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ODANetFast model
    model = ODANetFast().to(device)
    if not args.use_gt_depth:
        if os.path.exists(args.checkpoint):
            print(f"Loading Phase 4 Fast PyTorch weights from {args.checkpoint}...")
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Running with untrained weights.")
    model.eval()

    # JAX MPPI & Map
    print("Initializing JAX MPPI Controller and Sparse Map...")
    controller = JaxPAMPPIController(horizon=15, num_samples=10000, dt=0.05)
    
    # Calibrate Intrinsics (approximated from dataset camera)
    # VFOV=90° action camera on 224x224 → fx=fy=112
    sparse_map = RollingSparseMap(h=224, w=224, fx=151.7, fy=269.6, cx=112.0, cy=112.0, max_frames=30, points_per_frame=2048, voxel_size=0.15, max_unique=5000)

    # Load OptiTrack pose
    opti_path = os.path.join(os.path.dirname(args.video_path), "optitrack.csv")
    if os.path.exists(opti_path):
        df_opti = pd.read_csv(opti_path)
        opti_data = df_opti.to_numpy()
        R_calib = get_calib_rotation(opti_data)
    else:
        print(f"Error: Optitrack file {opti_path} not found.")
        return

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    start_p, start_R = get_drone_pose(opti_data, opti_data[0, 0] * 1e-9, R_calib)
    
    proj_distance = 50.0 
    forward_dir = start_R[:, 1]
    heading_dir = np.array([forward_dir[0], forward_dir[1], 0.0])
    norm = np.linalg.norm(heading_dir)
    if norm > 0: heading_dir /= norm
    else: heading_dir = np.array([0.0, 1.0, 0.0])
        
    final_p = start_p + proj_distance * heading_dir
    final_p[2] = start_p[2]
    goal_pos_jax = jnp.array(final_p, dtype=jnp.float32)

    collision_count = 0
    d_min_sum = 0.0
    frame_results = [] 
    prev_p_w = np.zeros(3, dtype=np.float32)

    frame_count = 0
    start_time = time.time()
    
    print(f"\nRunning JAX Fast Benchmark for Sequence {seq_id}...")
    pitch_deg = args.pitch_deg
    if pitch_deg is None:
        pitch_deg = calibrate_camera_pitch_from_first_frame(
            args.video_path, device, model, args.use_gt_depth, opti_data, R_calib, camera_vfov=45.1
        )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # State Estimation
        t_query = (frame_count - 1) / fps
        if opti_data is not None:
            p_w, R_w = get_drone_pose(opti_data, t_query, R_calib)
            if args.gimbal_mode:
                euler = Rotation.from_matrix(R_w).as_euler('xyz', degrees=False)
                R_w = Rotation.from_euler('xyz', [0.0, 0.0, euler[2]]).as_matrix().astype(np.float32)
            p_w_jax = jnp.array(p_w, dtype=jnp.float32)
            R_w_jax = jnp.array(R_w, dtype=jnp.float32)

        # Calculate velocity
        if frame_count == 1:
            vel_w = np.zeros(3, dtype=np.float32)
        else:
            vel_w = (p_w - prev_p_w) * fps
        prev_p_w = p_w.copy()

        # Perception
        if args.use_gt_depth:
            npy_path = os.path.join(os.path.dirname(args.video_path), "pseudo_depth", f"frame_{frame_count-1:04d}.npy")
            if os.path.exists(npy_path):
                depth_np = np.load(npy_path).astype(np.float32)
                gt_depth_raw = 1.0 / (depth_np / 255.0 + 0.1)
                gt_depth_raw = cv2.resize(gt_depth_raw, (224, 224), interpolation=cv2.INTER_NEAREST)
                depth_tensor = torch.from_numpy(gt_depth_raw).unsqueeze(0).unsqueeze(0).to(device)
                conf_tensor = torch.ones_like(depth_tensor)
            else:
                depth_tensor = torch.ones((1, 1, 224, 224), device=device) * 6.0
                conf_tensor = torch.zeros_like(depth_tensor)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            img_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                depth_tensor = outputs["depth"]
                conf_tensor = outputs["confidence"]

        # JAX Mapping
        obs_points_jax = sparse_map.update(
            depth_tensor, conf_tensor, 
            R_w_jax, p_w_jax, pitch_deg=pitch_deg, 
            conf_thresh=args.conf_thresh, max_depth=6.0
        )
        
        # Filter floor points (ignore points too close to ground z < 0.15)
        obs_points_jax = jnp.where(
            obs_points_jax[:, 2:3] <= 0.15,
            1e5,
            obs_points_jax
        )
        
        # MPPI
        # rpy from R_w
        yaw = Rotation.from_matrix(R_w).as_euler('xyz')[2]
        state_0_jax = jnp.array([p_w[0], p_w[1], p_w[2], vel_w[0], vel_w[1], vel_w[2], yaw], dtype=jnp.float32)
        
        action, best_traj, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, obs_points_jax)
        
        # TCR Eval
        best_traj_np = np.array(best_traj) # shape [Horizon, 4] -> [x, y, z, yaw]
        
        frame_min_dist = float('inf')
        for obs_gt in gt_obstacles:
            distances = np.linalg.norm(best_traj_np[:, 0:3] - obs_gt, axis=1)
            min_dist_to_obs = np.min(distances)
            if min_dist_to_obs < frame_min_dist:
                frame_min_dist = min_dist_to_obs
        
        obstacle_radius = 0.15  
        drone_radius = 0.15     
        surface_clearance = frame_min_dist - obstacle_radius - drone_radius
        
        status = "FAIL" if surface_clearance < 0.0 else "PASS"
        frame_results.append({
            "Frame": frame_count,
            "Clearance (m)": round(surface_clearance, 3),
            "Status": status
        })

        if surface_clearance < 0.0: collision_count += 1
        d_min_sum += surface_clearance

        if frame_count % 50 == 0:
            print(f"[{frame_count}/{total_frames}] TCR: {(collision_count/frame_count)*100:.2f}% | Avg Clearance: {d_min_sum/frame_count:.2f}m")

    cap.release()
    
    tcr_percent = (collision_count / frame_count) * 100.0 if frame_count > 0 else 0.0
    avg_dmin = d_min_sum / frame_count if frame_count > 0 else 0.0
    fps_val = frame_count / (time.time() - start_time)
    print(f"\n--- Benchmark Complete ---")
    print(f"TCR: {tcr_percent:.2f}% ({collision_count}/{frame_count} collisions)")
    print(f"Avg Surface Clearance (d_min): {avg_dmin:.3f}m")
    print(f"Inference FPS: {fps_val:.1f} Hz")
    
    os.makedirs("checkpoint/phase4", exist_ok=True)
    df_res = pd.DataFrame(frame_results)
    df_res.to_csv("checkpoint/phase4/tcr_results_fast.csv", index=False)
    print(f"Detailed results saved to checkpoint/phase4/tcr_results_fast.csv")

if __name__ == "__main__":
    main()
