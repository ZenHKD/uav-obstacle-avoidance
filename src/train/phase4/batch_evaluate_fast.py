import os
import json
import time
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from scipy.spatial.transform import Rotation
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.net_fast import ODANetFast
from src.control.jax_pa_mppi import JaxPAMPPIController
from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.train.phase4.oda_utils import get_drone_pose, get_calib_rotation, calibrate_camera_pitch_from_first_frame

import jax
import jax.numpy as jnp

def process_sequence(seq_id, args, model, controller, overview_df, device):
    video_path = os.path.join(args.dataset_dir, str(seq_id), f"{seq_id}.avi")
    opti_path = os.path.join(args.dataset_dir, str(seq_id), "optitrack.csv")
    
    if not os.path.exists(video_path) or not os.path.exists(opti_path):
        return None
        
    trial_rows = overview_df[overview_df['Sequence'] == seq_id]
    if len(trial_rows) == 0:
        return None

    gt_obstacles = []
    for _, row in trial_rows.iterrows():
        obs_x = float(row['Obstacle x'])
        obs_y = -float(row['Obstacle z'])
        obs_z = float(row['Obstacle y'])
        gt_obstacles.append(np.array([obs_x, obs_y, obs_z], dtype=np.float32))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    df_opti = pd.read_csv(opti_path)
    opti_data = df_opti.to_numpy()
    R_calib = get_calib_rotation(opti_data)
    start_p, start_R = get_drone_pose(opti_data, opti_data[0, 0] * 1e-9, R_calib)

    # Initialize JAX Sparse Map specific for this sequence
    sparse_map = RollingSparseMap(h=224, w=224, fx=151.7, fy=269.6, cx=112.0, cy=112.0, max_frames=30, points_per_frame=2048, voxel_size=0.15, max_unique=5000)

    proj_distance = 50.0  # Infinite horizon goal
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
    frame_count = 0
    prev_p_w = np.zeros(3, dtype=np.float32)
    
    pitch_deg = args.pitch_deg
    if pitch_deg is None:
        pitch_deg = calibrate_camera_pitch_from_first_frame(
            video_path, device, model, args.use_gt_depth, opti_data, R_calib, camera_vfov=45.1
        )
    print(f"[{seq_id}] Using camera pitch: {pitch_deg:.1f}°")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        t_query = (frame_count - 1) / fps
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

        if args.use_gt_depth:
            npy_path = os.path.join(args.dataset_dir, str(seq_id), "pseudo_depth", f"frame_{frame_count-1:04d}.npy")
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
        
        yaw = Rotation.from_matrix(R_w).as_euler('xyz')[2]
        state_0_jax = jnp.array([p_w[0], p_w[1], p_w[2], vel_w[0], vel_w[1], vel_w[2], yaw], dtype=jnp.float32)
        
        action, best_traj, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, obs_points_jax)
        
        best_traj_np = np.array(best_traj)
        frame_min_dist = float('inf')
        for obs_gt in gt_obstacles:
            distances = np.linalg.norm(best_traj_np[:, 0:3] - obs_gt, axis=1)
            min_dist_to_obs = np.min(distances)
            if min_dist_to_obs < frame_min_dist:
                frame_min_dist = min_dist_to_obs
        
        obstacle_radius = 0.15 
        drone_radius = 0.15    
        surface_clearance = frame_min_dist - obstacle_radius - drone_radius

        if surface_clearance < 0.0:
            collision_count += 1
            
        d_min_sum += surface_clearance

    cap.release()
    
    if frame_count == 0:
        return None
        
    return {
        "seq_id": seq_id,
        "frames": frame_count,
        "collisions": collision_count,
        "d_min_avg": d_min_sum / frame_count
    }

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluate Validation Sequences for JAX Fast Pipeline")
    parser.add_argument('--checkpoint', type=str, default="checkpoint/phase4/best_model_fast.pth")
    parser.add_argument('--val_json', type=str, default="checkpoint/phase4/val_sequences.json") # Reusing the dataset split
    parser.add_argument('--dataset_dir', type=str, default="dataset")
    parser.add_argument('--overview_csv', type=str, default="dataset/trial_overview.csv")
    parser.add_argument('--pitch_deg', type=float, default=None, help="Fixed camera mount pitch. If None, uses auto-calibration.")
    parser.add_argument('--gimbal_mode', action='store_true', help="Discard pitch/roll from OptiTrack and simulate a Gimbal")
    parser.add_argument('--conf_thresh', type=float, default=0.75)
    parser.add_argument('--use_gt_depth', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.val_json):
        # Fallback to phase 2 val sequences if phase 4 doesn't have it
        if args.val_json == "checkpoint/phase4/val_sequences.json":
            args.val_json = "checkpoint/phase2/val_sequences.json"
            
    if not os.path.exists(args.val_json):
        print(f"Error: Could not find {args.val_json}")
        return

    with open(args.val_json, 'r') as f:
        val_seqs = json.load(f)
        
    print(f"Loaded {len(val_seqs)} validation sequences from {args.val_json}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ODANetFast().to(device)
    if not args.use_gt_depth:
        if os.path.exists(args.checkpoint):
            print(f"Loading Model from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained weights.")
    model.eval()

    controller = JaxPAMPPIController(horizon=15, num_samples=10000, dt=0.05)
    overview_df = pd.read_csv(args.overview_csv)

    results = []
    total_frames = 0
    total_collisions = 0
    total_d_min_sum = 0.0

    print("\nStarting JAX Batch Evaluation...")
    pbar = tqdm(val_seqs)
    for seq_id in pbar:
        pbar.set_description(f"Processing Seq {seq_id}")
        res = process_sequence(seq_id, args, model, controller, overview_df, device)
        if res is not None:
            results.append(res)
            total_frames += res["frames"]
            total_collisions += res["collisions"]
            total_d_min_sum += res["d_min_avg"] * res["frames"]
            
            tcr = (res["collisions"] / res["frames"]) * 100
            pbar.set_postfix({"TCR": f"{tcr:.1f}%", "Clearance": f"{res['d_min_avg']:.2f}m"})

    if len(results) == 0:
        print("\nNo valid sequences were processed.")
        return

    df_results = pd.DataFrame(results)
    df_results["TCR (%)"] = (df_results["collisions"] / df_results["frames"] * 100).round(2)
    df_results["Clearance (m)"] = df_results["d_min_avg"].round(3)
    
    os.makedirs("checkpoint/phase4", exist_ok=True)
    out_csv = "checkpoint/phase4/val_batch_summary_fast.csv"
    
    df_results = df_results[["seq_id", "frames", "collisions", "TCR (%)", "Clearance (m)"]]
    df_results.to_csv(out_csv, index=False)
    
    global_tcr = (total_collisions / total_frames) * 100 if total_frames > 0 else 0
    global_d_min = total_d_min_sum / total_frames if total_frames > 0 else 0
    
    print("\n" + "="*60)
    print("--- GLOBAL FAST VALIDATION BENCHMARK SUMMARY ---")
    print("="*60)
    print(f"Sequences Evaluated: {len(results)} / {len(val_seqs)}")
    print(f"Total Frames:        {total_frames}")
    print(f"Total Collisions:    {total_collisions}")
    print(f"Global TCR:          {global_tcr:.2f} %")
    print(f"Global Avg Clearance:{global_d_min:.3f} m")
    print(f"\nDetailed batch results saved to: {out_csv}")
    print("="*60)

if __name__ == "__main__":
    main()
