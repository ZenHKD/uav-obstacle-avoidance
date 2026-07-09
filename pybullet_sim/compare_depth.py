import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
import pybullet as p
import importlib.util
from scipy.spatial.transform import Rotation
from transformers import pipeline
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.model.net_fast import ODANetFast
from pybullet_sim.env_generator import spawn_straight_racing_track

def get_camera_image(pos, rpy, client_id=0):
    yaw = rpy[2]
    rot = Rotation.from_euler('xyz', [0.0, 0.0, yaw])
    rot_mat = rot.as_matrix()
    forward = rot_mat[:, 1]
    up = np.array([0.0, 0.0, 1.0]) # Use true Z-up to prevent roll/pitch artifacts
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

def apply_colormap(depth_array, max_depth=10.0):
    depth_norm = np.clip(depth_array / max_depth, 0, 1)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    # Invert so close objects are bright (red/yellow), far are dark (blue)
    depth_norm = 255 - depth_norm
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_colored

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = ODANetFast(backbone_name='fastvit_t12').to(device)
    ckpt_path = "checkpoint/phase4/best_model_fast.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Warning: Model weights not found! Running with untrained weights.")
    model.eval()

    # Load HF Depth Anything
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    print(f"Loading Depth Anything Model on device: {device} with precision: {dtype}...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if device.type == 'cuda' else -1, dtype=dtype)

    # Init PyBullet
    INIT_XYZS = np.array([[0.0, -2.0, 1.0]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
    env = VelocityAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS, physics=Physics.PYB, pyb_freq=240, ctrl_freq=240, gui=False, record=False)
    PYB_CLIENT = env.getPyBulletClient()
    
    egl_spec = importlib.util.find_spec('eglRenderer')
    if egl_spec and egl_spec.origin:
        p.loadPlugin(egl_spec.origin, "_eglRendererPlugin", physicsClientId=PYB_CLIENT)

    obs_dict, info = env.reset()
    spawn_straight_racing_track(client_id=PYB_CLIENT)

    out_dir = "pybullet_sim/depth_comparison"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    images_to_concat = []
    
    frame_count = 0
    capture_count = 0
    
    print("Flying forward and capturing 8 frames...")
    
    while capture_count < 8:
        # Teleport drone forward instead of relying on un-tuned PID physics
        current_y = -2.0 + capture_count * 0.5 # Move 0.5 meters forward each capture
        new_pos = [0.0, current_y, 1.0]
        new_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], new_pos, new_quat, physicsClientId=PYB_CLIENT)
        
        # Step simulation to update camera sensors
        p.stepSimulation(physicsClientId=PYB_CLIENT)
        
        pos = np.array(new_pos)
        rpy = np.array([0.0, 0.0, 0.0])
        
        rgb_fpv, depth_gt = get_camera_image(pos, rpy, PYB_CLIENT)
        rgb_input = rgb_fpv[:, :, :3]
            
        # Inference
        img_tensor = torch.from_numpy(rgb_input.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            depth_pred = outputs["depth"][0, 0].cpu().numpy()
            
        # Depth Anything Inference
        pil_image = Image.fromarray(rgb_input)
        hf_res = pipe(pil_image)
        depth_hf_raw = np.array(hf_res["depth"]).astype(np.float32)
        # Convert HF relative depth (0-255, 255=close) to rough metric depth (0-10, 0=close)
        depth_hf_metric = 10.0 * (1.0 - depth_hf_raw / 255.0)
            
        # Fair Error Calculation (capped at 10.0m for full simulation horizon)
        valid_mask = depth_gt > 0.0
        gt_eval = np.clip(depth_gt[valid_mask], 0.0, 10.0)
        oda_eval = np.clip(depth_pred[valid_mask], 0.0, 10.0)
        hf_eval = np.clip(depth_hf_metric[valid_mask], 0.0, 10.0)
        
        error_map = np.abs(gt_eval - oda_eval)
        mae = np.mean(error_map)
        rmse = np.sqrt(np.mean(error_map**2))
        
        hf_error_map = np.abs(gt_eval - hf_eval)
        hf_mae = np.mean(hf_error_map)
        hf_rmse = np.sqrt(np.mean(hf_error_map**2))
        
        results.append({
            "Capture_ID": capture_count + 1,
            "ODA_MAE": mae,
            "ODA_RMSE": rmse,
            "HF_MAE": hf_mae,
            "HF_RMSE": hf_rmse
        })
        
        # Format Images
        rgb_bgr = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2BGR)
        gt_colored = apply_colormap(depth_gt)
        pred_colored = apply_colormap(depth_pred)
        hf_colored = apply_colormap(depth_hf_metric)
        
        # Add text labels
        cv2.putText(rgb_bgr, f"RGB (ID: {capture_count+1})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(gt_colored, f"GT Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pred_colored, f"ODANet (MAE: {mae:.2f}m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(hf_colored, f"DA2 (MAE: {hf_mae:.2f}m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Horizontal concat: [RGB | GT | ODANet | DA2]
        row_img = np.hstack((rgb_bgr, gt_colored, pred_colored, hf_colored))
        images_to_concat.append(row_img)
        
        print(f"Captured {capture_count+1}/8 | ODANet MAE: {mae:.3f}m | DA2 MAE: {hf_mae:.3f}m")
        capture_count += 1
            
    env.close()
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "depth_error_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nStats saved to {csv_path}")
    
    # Save Grid Image
    final_img = np.vstack(images_to_concat)
    img_path = os.path.join(out_dir, "depth_comparison_grid.png")
    cv2.imwrite(img_path, final_img)
    print(f"Grid image saved to {img_path}")

if __name__ == "__main__":
    main()
