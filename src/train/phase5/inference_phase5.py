import os
import time
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from scipy.spatial.transform import Rotation

# Force Matplotlib to use a headless backend (Agg) BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.net_fast import ODANetFast
from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.control.jax_pa_mppi import JaxPAMPPIController
from src.train.phase4.oda_utils import get_drone_pose, get_calib_rotation, calibrate_camera_pitch_from_first_frame





def save_interactive_world_map_with_mppi(trajectory, valid_points_list, mppi_history, output_html_path, dynamic_goal_pos=None, gt_obstacles=None):
    import plotly.graph_objects as go
    
    data = []
    
    # 0. Recorded UAV Path (Trace 0)
    traj_np = np.round(np.array(trajectory), 2)
    data.append(go.Scatter3d(
        x=traj_np[:, 0],
        y=traj_np[:, 1],
        z=traj_np[:, 2],
        mode='lines+markers',
        name='Recorded UAV Path',
        line=dict(color='blue', width=4),
        marker=dict(size=2, color='darkblue'),
        showlegend=True
    ))
    
    # 1. Target Position (Trace 1)
    if dynamic_goal_pos is not None:
        goal_x, goal_y, goal_z = dynamic_goal_pos
    else:
        goal_x, goal_y, goal_z = traj_np[-1, 0], traj_np[-1, 1], traj_np[-1, 2]
        
    data.append(go.Scatter3d(
        x=[goal_x], y=[goal_y], z=[goal_z],
        mode='markers', name='End Goal',
        marker=dict(size=8, color='gold', symbol='diamond'),
        showlegend=True
    ))

    # 2. Straight-Line Projected Path (Trace 2)
    if dynamic_goal_pos is not None and len(traj_np) > 0:
        data.append(go.Scatter3d(
            x=[traj_np[0, 0], goal_x], y=[traj_np[0, 1], goal_y], z=[traj_np[0, 2], goal_z],
            mode='lines', name='Straight-Line Projected Path',
            line=dict(color='orange', width=3, dash='dash'),
            showlegend=True
        ))
    else:
        data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Straight-Line Projected Path'))

    first_step = mppi_history[0] if len(mppi_history) > 0 else None
    
    # 3. Dynamic Sparse Point Cloud (Trace 3)
    first_pts = first_step['known_points'] if first_step is not None else []
    if len(first_pts) > 0:
        first_pts_np = np.array(first_pts)
        data.append(go.Scatter3d(
            x=first_pts_np[:, 0], y=first_pts_np[:, 1], z=first_pts_np[:, 2],
            mode='markers', name='Obstacles (Sparse PC)',
            marker=dict(size=2, color='red', opacity=0.8),
            showlegend=True
        ))
    else:
        data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Obstacles (Sparse PC)'))
        
    # 4. Dynamic UAV Position (Trace 4)
    init_p_w = np.round(first_step['p_w'] if first_step is not None else traj_np[0], 2)
    data.append(go.Scatter3d(
        x=[init_p_w[0]], y=[init_p_w[1]], z=[init_p_w[2]],
        mode='markers', name='Current Position',
        marker=dict(size=8, color='cyan', symbol='circle', line=dict(color='black', width=2)),
        showlegend=True
    ))
    
    # 5. Dynamic Best Path (Trace 5)
    if first_step is not None:
        best_p = np.round(first_step['best_path'], 2)
        data.append(go.Scatter3d(
            x=best_p[:, 0], y=best_p[:, 1], z=best_p[:, 2],
            mode='lines', name='PA-MPPI Planned Path',
            line=dict(color='green', width=5),
            showlegend=True
        ))
    else:
        data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='PA-MPPI Planned Path'))
        
    # 6. Dynamic Candidate Paths (Trace 6)
    if first_step is not None:
        cand_x, cand_y, cand_z = [], [], []
        for p in first_step['candidate_paths']:
            cand_x.extend(np.round(p[:, 0], 2).tolist() + [None])
            cand_y.extend(np.round(p[:, 1], 2).tolist() + [None])
            cand_z.extend(np.round(p[:, 2], 2).tolist() + [None])
        data.append(go.Scatter3d(
            x=cand_x, y=cand_y, z=cand_z,
            mode='lines', name='PA-MPPI Candidates',
            line=dict(color='purple', width=1.5),
            opacity=0.15,
            showlegend=True
        ))
    else:
        data.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='PA-MPPI Candidates'))

    # 7. Ground Truth Obstacles — 3D Cylinders
    if gt_obstacles is not None and len(gt_obstacles) > 0:
        for i, obs in enumerate(gt_obstacles):
            ox, oy, oz = float(obs[0]), float(obs[1]), float(obs[2])
            radius = 0.15  # obstacle column radius (m)
            n_pts = 20
            theta = np.linspace(0, 2 * np.pi, n_pts)
            z_levels = np.linspace(0, 2.0, 10)  # cylinder height: 0 to 2m
            # Build cylinder mesh vertices
            cyl_x, cyl_y, cyl_z = [], [], []
            cyl_i, cyl_j, cyl_k = [], [], []
            for zl_idx, zl in enumerate(z_levels):
                for t_idx in range(n_pts):
                    cyl_x.append(ox + radius * np.cos(theta[t_idx]))
                    cyl_y.append(oy + radius * np.sin(theta[t_idx]))
                    cyl_z.append(zl)
            # Build faces between adjacent z-levels
            for zl_idx in range(len(z_levels) - 1):
                base = zl_idx * n_pts
                top = (zl_idx + 1) * n_pts
                for t_idx in range(n_pts):
                    next_t = (t_idx + 1) % n_pts
                    cyl_i.extend([base + t_idx, base + t_idx])
                    cyl_j.extend([base + next_t, top + t_idx])
                    cyl_k.extend([top + t_idx, top + next_t])
            name = f'GT Obstacle {i+1} ({ox:.2f}, {oy:.2f})'
            data.append(go.Mesh3d(
                x=cyl_x, y=cyl_y, z=cyl_z,
                i=cyl_i, j=cyl_j, k=cyl_k,
                color='orangered', opacity=0.4,
                name=name, showlegend=True,
                hoverinfo='name'
            ))

    steps = []
    for step in mppi_history:
        f_idx = step['frame_idx']
        p_w = step['p_w']
        best_p = step['best_path']
        cands = step['candidate_paths']
        step_pts = np.array(step['known_points'])
        
        if len(step_pts) > 0:
            vox_x = step_pts[:, 0].tolist()
            vox_y = step_pts[:, 1].tolist()
            vox_z = step_pts[:, 2].tolist()
        else:
            vox_x, vox_y, vox_z = [], [], []
            
        cand_x, cand_y, cand_z = [], [], []
        for p in cands:
            cand_x.extend(np.round(p[:, 0], 2).tolist() + [None])
            cand_y.extend(np.round(p[:, 1], 2).tolist() + [None])
            cand_z.extend(np.round(p[:, 2], 2).tolist() + [None])
            
        best_x = np.round(best_p[:, 0], 2).tolist()
        best_y = np.round(best_p[:, 1], 2).tolist()
        best_z = np.round(best_p[:, 2], 2).tolist()
        
        slider_step = dict(
            args=[
                {
                    'x': [vox_x, [float(np.round(p_w[0], 2))], best_x, cand_x],
                    'y': [vox_y, [float(np.round(p_w[1], 2))], best_y, cand_y],
                    'z': [vox_z, [float(np.round(p_w[2], 2))], best_z, cand_z]
                },
                [3, 4, 5, 6]
            ],
            label=f"F {f_idx}",
            method='restyle'
        )
        steps.append(slider_step)
        
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Timeline Frame: ", font=dict(size=16), visible=True),
        pad=dict(t=50),
        steps=steps
    )]
    
    # Compute dynamic axis range from trajectory + obstacles
    all_x = traj_np[:, 0].tolist()
    all_y = traj_np[:, 1].tolist()
    if gt_obstacles is not None:
        for obs in gt_obstacles:
            all_x.append(float(obs[0]))
            all_y.append(float(obs[1]))
    x_range = [min(all_x) - 2, max(all_x) + 2]
    y_range = [min(all_y) - 2, max(all_y) + 2]

    layout = go.Layout(
        title=dict(text="PA-MPPI JAX Trajectory Planning Evolution (with GT Obstacles)", font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='X (Right)', range=x_range),
            yaxis=dict(title='Y (Forward)', range=y_range),
            zaxis=dict(title='Z (Up)', range=[-0.5, 2.5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.25, z=0.6)
        ),
        sliders=sliders,
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(output_html_path)
    print(f"Interactive HTML map saved to {output_html_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="checkpoint/phase4/best_model_fast.pth", help="Path to ODANet model checkpoint")
    parser.add_argument('--video_path', type=str, default="dataset/11/11.avi", help="Path to input video")
    parser.add_argument('--output_path', type=str, default="checkpoint/phase5/inference_phase5_jax_demo.mp4")
    parser.add_argument('--conf_thresh', type=float, default=0.75, help="Confidence threshold to penalize virtual voxels")
    parser.add_argument('--use_gt_depth', action='store_true', help="Use Ground Truth depth instead of network predictions")
    parser.add_argument('--no_render', action='store_true', help="Disable Matplotlib 3D rendering to boost FPS")
    parser.add_argument('--overview_csv', type=str, default="dataset/trial_overview.csv", help="Path to trial_overview.csv for GT obstacles")
    parser.add_argument('--pitch_deg', type=float, default=None, help="Fixed camera mount pitch in degrees. If None, uses auto-calibration.")
    parser.add_argument('--gimbal_mode', action='store_true', help="Discard pitch/roll from OptiTrack and simulate a Gimbal (like PyBullet)")
    args = parser.parse_args()

    opti_path = os.path.join(os.path.dirname(args.video_path), "optitrack.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load Fast ODANet model
    print(f"Loading Model from {args.checkpoint}...")
    model = ODANetFast(
        backbone_name="fastvit_t8",
        pretrained=False
    ).to(device)
    
    # Lấy state_dict, filter out projection keys nếu có
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_state = checkpoint['model_state_dict']
    filtered_state = {k: v for k, v in model_state.items() if not k.startswith("projection")}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    print("Model loaded successfully!")

    # Instantiate JAX Geometry and Controller
    print("Initializing JAX Geometry and Controller...")
    H, W = 224, 224
    # Original camera: 1920x1080, fx=fy=1300, cx=960, cy=540
    # When resized to 224x224 (non-aspect-ratio-preserving):
    #   fx_224 = 1300 * (224/1920), fy_224 = 1300 * (224/1080)
    fx = 1300.0 * (224.0 / 1920.0)   # ≈ 151.7
    fy = 1300.0 * (224.0 / 1080.0)   # ≈ 269.6
    cx = 960.0 * (224.0 / 1920.0)    # = 112.0
    cy = 540.0 * (224.0 / 1080.0)    # = 112.0
    
    sparse_map = RollingSparseMap(
        h=H, w=W, fx=fx, fy=fy, cx=cx, cy=cy,
        max_frames=30, points_per_frame=2048, voxel_size=0.15, max_unique=5000
    )
    
    controller = JaxPAMPPIController(horizon=15, num_samples=10000, dt=0.05)

    # Load OptiTrack pose
    opti_data = None
    R_calib = None
    if os.path.exists(opti_path):
        print("Loading OptiTrack pose data...")
        df = pd.read_csv(opti_path)
        opti_data = df.to_numpy()
        R_calib = get_calib_rotation(opti_data)

    # Load video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    pitch_deg = args.pitch_deg
    if pitch_deg is None:
        pitch_deg = calibrate_camera_pitch_from_first_frame(
            args.video_path, device, model, args.use_gt_depth, opti_data, R_calib, camera_vfov=45.1
        )
    print(f"Using camera pitch: {pitch_deg:.1f}°")

    # Load ground truth obstacles from trial_overview.csv
    gt_obstacles = []
    seq_id_str = os.path.basename(os.path.dirname(args.video_path))
    try:
        seq_id = int(seq_id_str)
        overview_df = pd.read_csv(args.overview_csv)
        trial_rows = overview_df[overview_df['Sequence'] == seq_id]
        P_conv = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)  # OptiTrack → robotics
        for _, row in trial_rows.iterrows():
            obs_x = float(row['Obstacle x'])
            obs_y = -float(row['Obstacle z'])   # OptiTrack z → robotics -y
            obs_z = float(row['Obstacle y'])     # OptiTrack y(height) → robotics z
            gt_obstacles.append(np.array([obs_x, obs_y, obs_z], dtype=np.float32))
        if len(gt_obstacles) > 0:
            print(f"Loaded {len(gt_obstacles)} GT obstacle(s) for sequence {seq_id}")
            for i, obs in enumerate(gt_obstacles):
                print(f"  Obstacle {i+1}: x={obs[0]:.3f}, y={obs[1]:.3f}, z={obs[2]:.3f}")
    except (ValueError, FileNotFoundError) as e:
        print(f"[GT Obstacles] Could not load: {e}")

    # Determine straight-line projection goal from initial pose
    goal_pos = np.zeros(3, dtype=np.float32)
    proj_distance = 50.0  # Infinite horizon
    if opti_data is not None:
        start_p, start_R = get_drone_pose(opti_data, opti_data[0, 0] * 1e-9)
        forward_dir = start_R[:, 1]
        heading_dir = np.array([forward_dir[0], forward_dir[1], 0.0])
        norm = np.linalg.norm(heading_dir)
        if norm > 0: heading_dir /= norm
        else: heading_dir = np.array([0.0, 1.0, 0.0])
            
        final_p = start_p + proj_distance * heading_dir
        final_p[2] = start_p[2]
        goal_pos = final_p
        print(f"Start pos: {start_p}, Forward dir: {heading_dir}")
        print(f"Dynamic Straight-Line Goal set to: {final_p}")

    # Figure initialization for video output
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')
    canvas = FigureCanvas(fig)

    out_w, out_h = 1920, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.output_path, fourcc, fps, (out_w, out_h))

    trajectory = []
    mppi_history = []
    prev_p_w = np.zeros(3, dtype=np.float32)
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        total_time = time.time() - start_time
        current_fps = frame_count / (total_time + 1e-9)

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        img_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            depth_tensor = outputs["depth"]
            conf_tensor = outputs["confidence"]
            pred_d = depth_tensor[0, 0].cpu().numpy()

            # Drone State Estimation
            t_query = (frame_count - 1) / fps
            p_w = np.zeros(3, dtype=np.float32)
            R_w = np.eye(3, dtype=np.float32)
            euler = [0.0, 0.0, 0.0]
            
            if opti_data is not None:
                p_w, R_w = get_drone_pose(opti_data, t_query, R_calib)
                trajectory.append(p_w)
                if args.gimbal_mode:
                    # Match PyBullet closed-loop: extract yaw only
                    euler = Rotation.from_matrix(R_w).as_euler('xyz', degrees=False)
                    R_w = Rotation.from_euler('xyz', [0.0, 0.0, euler[2]]).as_matrix().astype(np.float32)
                else:
                    euler = Rotation.from_matrix(R_w).as_euler('xyz')
                
            # Calculate velocity
            if frame_count == 1:
                vel_w = np.zeros(3, dtype=np.float32)
            else:
                vel_w = (p_w - prev_p_w) * fps
            prev_p_w = p_w.copy()

            # 1. Update Geometry (JAX Rolling Buffer)
            import jax.numpy as jnp
            R_w_jax = jnp.array(R_w)
            p_w_jax = jnp.array(p_w)
            
            # Extract Global Point Cloud from JAX
            unique_points_jax = sparse_map.update(
                depth_tensor, conf_tensor, 
                R_w_jax, p_w_jax, pitch_deg, 
                conf_thresh=args.conf_thresh, max_depth=6.0
            )

            # Filter floor points (ignore points too close to ground z < 0.15)
            unique_points_jax = jnp.where(
                unique_points_jax[:, 2:3] <= 0.15,
                1e5,
                unique_points_jax
            )

            # 2. Run PA-MPPI Control (JAX)
            state_0 = [p_w[0], p_w[1], p_w[2], vel_w[0], vel_w[1], vel_w[2], euler[2]] # [px, py, pz, vx, vy, vz, yaw]
            
            action_jax, best_traj_jax, cands_jax, costs_jax = controller.compute_action(
                state_0=state_0,
                goal_pos=goal_pos,
                obs_points=unique_points_jax
            )

            # Convert outputs to numpy for visualization
            best_traj = np.array(best_traj_jax)
            cands_np = np.array(cands_jax)
            costs_np = np.array(costs_jax)
            unique_points_np = np.array(unique_points_jax)
            
            # Filter out garbage points (1e5) for visualization
            valid_mask = unique_points_np[:, 2] < 50000.0
            valid_points = unique_points_np[valid_mask]

            if frame_count % 5 == 1:
                # Select top 30 candidates for HTML plotting
                top_cand_indices = np.argsort(costs_np)[:30]
                top_cands = cands_np[top_cand_indices][:, :, :3]
                
                # Further subsample points for HTML size
                if len(valid_points) > 1000:
                    sub_idx = np.random.choice(len(valid_points), 1000, replace=False)
                    html_points = valid_points[sub_idx]
                else:
                    html_points = valid_points
                    
                mppi_history.append({
                    'frame_idx': frame_count,
                    'p_w': p_w.copy(),
                    'best_path': best_traj[:, :3],
                    'candidate_paths': top_cands,
                    'known_points': html_points.tolist()
                })

            if not args.no_render:
                # Draw plotting outputs
                ax1.clear(); ax2.clear(); ax3.clear()
                ax1.axis('off'); ax2.axis('off')
                ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

                # 1. Plot RGB
                ax1.imshow(frame_resized)
                ax1.set_title(f"Camera Input (FPS: {current_fps:.1f})")

                # 2. Plot Depth
                ax2.imshow(pred_d, cmap="inferno", vmin=0.0, vmax=6.0)
                ax2.set_title("Predicted Metric Depth")

                # 3. Plot 3D Environment
                if len(trajectory) > 0:
                    traj_np = np.array(trajectory)
                    ax3.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], color='blue', linewidth=2)
                    ax3.scatter([p_w[0]], [p_w[1]], [p_w[2]], color='cyan', marker='o', s=80, edgecolors='black')

                # Draw GT Obstacle Cylinders
                for obs in gt_obstacles:
                    theta_cyl = np.linspace(0, 2*np.pi, 20)
                    z_cyl = np.linspace(0, 2.0, 10)
                    theta_g, z_g = np.meshgrid(theta_cyl, z_cyl)
                    x_cyl = 0.15 * np.cos(theta_g) + obs[0]
                    y_cyl = 0.15 * np.sin(theta_g) + obs[1]
                    ax3.plot_surface(x_cyl, y_cyl, z_g, alpha=0.3, color='orange')

                # Draw Point Cloud
                if len(valid_points) > 0:
                    ax3.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                                c='red', marker='.', s=5, alpha=0.5)

                # Draw best trajectory
                ax3.plot(best_traj[:, 0], best_traj[:, 1], best_traj[:, 2], color='green', linewidth=3)

                # Draw candidate trajectories
                top_indices = np.argsort(costs_np)[:30]
                for t_idx in top_indices:
                    tc = cands_np[t_idx]
                    ax3.plot(tc[:, 0], tc[:, 1], tc[:, 2], color='purple', alpha=0.1, linewidth=1)

                ax3.set_title("JAX PA-MPPI Trajectories")
                # Dynamic axis limits centered on drone position
                view_radius = 6.0  # meters around drone to show
                ax3.set_xlim(p_w[0] - view_radius, p_w[0] + view_radius)
                ax3.set_ylim(p_w[1] - 2.0, p_w[1] + view_radius * 1.5)
                ax3.set_zlim(-0.5, 3.0)
                ax3.view_init(elev=30, azim=-60)
                plt.tight_layout()

                canvas.draw()
                buf = canvas.buffer_rgba()
                img_bgr = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2BGR)
                out_vid.write(cv2.resize(img_bgr, (out_w, out_h)))

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames... Avg FPS: {current_fps:.1f}")

    cap.release()
    out_vid.release()
    plt.close(fig)
    print(f"\nPhase 3 JAX MPPI Video saved to {args.output_path}")

    # Generate HTML
    html_output_path = args.output_path.replace('.mp4', '.html')
    save_interactive_world_map_with_mppi(trajectory, [], mppi_history, html_output_path, dynamic_goal_pos=goal_pos, gt_obstacles=gt_obstacles)

if __name__ == '__main__':
    main()
