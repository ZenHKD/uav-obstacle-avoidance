"""
Shared utility functions for ODA dataset evaluation.

Uses R_calib to align body frame gravity direction with Z-up,
then calibrate_camera_pitch to determine the camera mount angle.
"""

import os
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation

# OptiTrack world → Robotics world: x_r = x_o, y_r = -z_o, z_r = y_o
P_MAT = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)


def get_calib_rotation(opti_data):
    """
    Compute a static calibration rotation R_calib that maps the arbitrary
    OptiTrack rigid body frame to the true Robot Body frame (X=Right, Y=Forward, Z=Up).
    
    This fixes both the arbitrary Up axis (pitch/roll offset) and the arbitrary 
    Forward axis (yaw offset) introduced when the rigid body was created in Motive.
    """
    quats = opti_data[:, 4:8]
    g_raw = np.array([0, 1, 0], dtype=np.float32)  # gravity direction in OptiTrack world (Y-up)
    
    # 1. Z-axis: Average gravity direction in OptiTrack body frame
    g_body_raws = []
    for q in quats:
        r = Rotation.from_quat(q).as_matrix()
        g_body_raws.append(r.T @ g_raw)
    
    g_body_raw_mean = np.mean(g_body_raws, axis=0)
    Z_basis = g_body_raw_mean / (np.linalg.norm(g_body_raw_mean) + 1e-9)
    
    # 2. Y-axis: Flight direction in Robot World -> OptiTrack Body frame
    pos = opti_data[:, 1:4]
    p_start_w = np.array([pos[0, 0], -pos[0, 2], pos[0, 1]], dtype=np.float32)
    p_end_w = np.array([pos[-1, 0], -pos[-1, 2], pos[-1, 1]], dtype=np.float32)
    
    v_flight = p_end_w[0:2] - p_start_w[0:2]
    if np.linalg.norm(v_flight) < 1e-5:
        # Fallback if drone doesn't move
        v_flight = np.array([0.0, 1.0], dtype=np.float32)
    else:
        v_flight = v_flight / np.linalg.norm(v_flight)
        
    v_flight_robot = np.array([v_flight[0], v_flight[1], 0.0], dtype=np.float32)
    
    # P_MAT maps OptiTrack World -> Robot World
    P_MAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    r_start = Rotation.from_quat(quats[0]).as_matrix()
    
    # Map the forward flight vector back to the OptiTrack Body frame at start
    f_body = r_start.T @ P_MAT.T @ v_flight_robot
    
    # 3. Construct Orthogonal Basis (X, Y, Z)
    X_basis = np.cross(f_body, Z_basis)
    if np.linalg.norm(X_basis) < 1e-5:
        X_basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        X_basis /= np.linalg.norm(X_basis)
        
    Y_basis = np.cross(Z_basis, X_basis)
    Y_basis /= np.linalg.norm(Y_basis)
    
    # R_calib.T maps from Robot Body to OptiTrack Body, so columns are the basis vectors
    R_calib_T = np.column_stack([X_basis, Y_basis, Z_basis])
    R_calib = R_calib_T.T
    
    return R_calib.astype(np.float32)


def get_drone_pose(opti_data, t_query_sec, R_calib=None):
    """
    Get interpolated drone position and rotation at query time.
    
    Returns (p_robot, R_robot) in world frame (X-right, Y-forward, Z-up).
    R_robot columns: [right, forward, up]
    
    If R_calib is provided, applies the calibration rotation.
    """
    times = opti_data[:, 0] * 1e-9
    pos = opti_data[:, 1:4]
    quats = opti_data[:, 4:8]
    
    t_query = np.clip(t_query_sec, times[0], times[-1])
    idx = np.searchsorted(times, t_query)
    
    if idx == 0:
        p = pos[0]
        q = quats[0]
    elif idx >= len(times):
        p = pos[-1]
        q = quats[-1]
    else:
        t0 = times[idx - 1]
        t1 = times[idx]
        alpha = (t_query - t0) / (t1 - t0 + 1e-9)
        p = (1.0 - alpha) * pos[idx - 1] + alpha * pos[idx]
        q0 = quats[idx - 1]
        q1 = quats[idx]
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        q = (1.0 - alpha) * q0 + alpha * q1
        q /= (np.linalg.norm(q) + 1e-9)
    
    # Position: OptiTrack [x, y_height, z] → Robot [x, -z, y_height]
    p_robot = np.array([p[0], -p[2], p[1]], dtype=np.float32)
    
    # Rotation
    r_raw = Rotation.from_quat(q).as_matrix().astype(np.float32)
    if R_calib is not None:
        R_robot = P_MAT @ r_raw @ R_calib.T
    else:
        # Fallback: direct column swap (less accurate but works without calibration)
        R_world = P_MAT @ r_raw
        R_robot = np.column_stack([-R_world[:, 0], R_world[:, 2], R_world[:, 1]])
    
    return p_robot, R_robot


def calibrate_camera_pitch_from_first_frame(video_path, device, model=None, 
                                              use_gt_depth=False, opti_data=None,
                                              R_calib=None, camera_vfov=90.0):
    """
    Calibrate camera mount pitch angle from the first frame of a video.
    
    Uses OptiTrack drone height + depth model to compute:
        mount_angle = arcsin(height / depth_ground) - VFOV/2
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[PitchCalib] Warning: Could not open {video_path}, defaulting to 0.0 deg")
        return 0.0
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return 0.0
    
    if use_gt_depth:
        seq_dir = os.path.dirname(video_path)
        npy_path = os.path.join(seq_dir, "pseudo_depth", "frame_0000.npy")
        if os.path.exists(npy_path):
            depth_np = np.load(npy_path).astype(np.float32)
            depth_raw = 1.0 / (depth_np / 255.0 + 0.1)
            depth_map = cv2.resize(depth_raw, (224, 224), interpolation=cv2.INTER_NEAREST)
        else:
            print(f"[PitchCalib] Warning: GT depth not found at {npy_path}, defaulting to 0.0 deg")
            return 0.0
    else:
        if model is None:
            return 0.0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        img_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            depth_map = outputs["depth"][0, 0].cpu().numpy()
    
    h = depth_map.shape[0]
    # Ground depth: bottom 10% of image (most likely ground pixels)
    ground_depth = depth_map[int(h * 0.9):, :].mean()
    
    # === Geometric calibration using OptiTrack height ===
    if opti_data is not None:
        if R_calib is not None:
            p_w, _ = get_drone_pose(opti_data, opti_data[0, 0] * 1e-9, R_calib)
        else:
            p_w, _ = get_drone_pose(opti_data, opti_data[0, 0] * 1e-9)
        drone_height = abs(p_w[2])
        
        if drone_height > 0.3 and ground_depth > 0.3:
            sin_val = np.clip(drone_height / ground_depth, 0.0, 1.0)
            total_angle = np.degrees(np.arcsin(sin_val))
            pitch_deg = -(total_angle - camera_vfov / 2.0)
            pitch_deg = float(np.clip(pitch_deg, -45.0, 0.0))
            print(f"[PitchCalib] Geometric: height={drone_height:.3f}m, ground_depth={ground_depth:.2f}m, "
                  f"VFOV={camera_vfov:.0f}° → mount pitch: {pitch_deg:.1f}°")
            return pitch_deg
    
    # === Fallback — depth gradient heuristic ===
    top_band = depth_map[:h // 4, :].mean()
    bottom_band = depth_map[3 * h // 4:, :].mean()
    
    if bottom_band < top_band * 0.7:
        ratio = top_band / (bottom_band + 1e-6)
        log_ratio = np.log(max(ratio, 1.0))
        pitch_deg = -log_ratio * 10.0
        pitch_deg = float(np.clip(pitch_deg, -30.0, 0.0))
    else:
        pitch_deg = 0.0
    
    print(f"[PitchCalib] Fallback: top={top_band:.2f} bot={bottom_band:.2f} → pitch: {pitch_deg:.1f}°")
    return pitch_deg
