import os
import csv
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy.spatial.transform import Rotation

class ODADataset(Dataset):
    def __init__(self, dataset_dir="dataset", transform=None, target_size=(224, 224), cache_data=True):
        """
        The Obstacle Detection & Avoidance (ODA) Dataset loader.
        Frame-level dataset (no sequences, no IMU).
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_size = target_size
        self.cache_data = cache_data
        
        # Camera intrinsics
        self.fx, self.fy = 1300.0, 1300.0
        self.cx, self.cy = 960.0, 540.0
        
        self.fx_s = self.fx * (target_size[1] / 1920.0)
        self.fy_s = self.fy * (target_size[0] / 1080.0)
        self.cx_s = self.cx * (target_size[1] / 1920.0)
        self.cy_s = self.cy * (target_size[0] / 1080.0)
        
        self.R_c_to_b = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        self.P = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)
        
        self.optitrack_cache = {}
        
        self.trial_overview = self._parse_trial_overview()
        self.samples = []
        self._build_samples()
        
    def _parse_trial_overview(self):
        csv_path = os.path.join(self.dataset_dir, "trial_overview.csv")
        overview = {}
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Overview file not found: {csv_path}")
            
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            seq_idx = header.index('Sequence')
            lux_idx = header.index('Lux')
            incl_idx = header.index('Incl. Video')
            
            for row in reader:
                if not row: continue
                seq_id = int(row[seq_idx])
                
                # Skip known bad sequences
                if seq_id in [1306, 1321, 1344] or (593 <= seq_id <= 629):
                    continue
                    
                lux = int(row[lux_idx])
                incl_video = int(row[incl_idx])
                
                if incl_video != 1:
                    continue
                    
                trial_path = os.path.join(self.dataset_dir, str(seq_id))
                video_path = os.path.join(trial_path, f"{seq_id}.avi")
                pseudo_depth_dir = os.path.join(trial_path, "pseudo_depth")
                
                if not os.path.exists(video_path) or not os.path.exists(pseudo_depth_dir):
                    continue
                    
                overview[seq_id] = {
                    'lux': lux,
                    'video_path': video_path,
                    'pseudo_depth_dir': pseudo_depth_dir
                }
                
        # Fast parsing: infer frame count directly from .npy count (bypassing slow cv2 video loading)
        for seq_id, info in list(overview.items()):
            try:
                npy_files = [f for f in os.listdir(info['pseudo_depth_dir']) if f.endswith('.npy')]
                if not npy_files:
                    overview.pop(seq_id)
                    continue
                info['frame_count'] = len(npy_files)
            except Exception:
                overview.pop(seq_id)
            
        return overview
        
    def _build_samples(self):
        for seq_id, info in self.trial_overview.items():
            frame_count = info['frame_count']
            # Only use every 5th frame to avoid highly redundant consecutive frames in train set
            for frame_idx in range(0, frame_count, 5):
                # Verify that the pseudo depth frame exists
                npy_path = os.path.join(info['pseudo_depth_dir'], f"frame_{frame_idx:04d}.npy")
                if os.path.exists(npy_path):
                    self.samples.append({
                        'seq_id': seq_id,
                        'frame': frame_idx,
                        'lux': info['lux'],
                        'npy_path': npy_path
                    })

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_id = sample['seq_id']
        f = sample['frame']
        info = self.trial_overview[seq_id]
        
        # 1. Load RGB frame
        cap = cv2.VideoCapture(info['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            rgb_tensor = self.transform(frame_rgb)
        else:
            rgb_np = frame_rgb.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1) # (3, H, W)
            
        # 2. Load Distilled Pseudo Depth
        depth_np = np.load(sample['npy_path']).astype(np.float32)
        depth_map = 1.0 / (depth_np / 255.0 + 0.1) 
        depth_tensor = torch.from_numpy(depth_map)
        
        # 3. Compute Distance Ground Truth from Pseudo Depth (Bypassing OptiTrack bugs completely!)
        # Extract the center 50% of the image to find distance to nearest forward obstacle
        h, w = depth_map.shape
        center_crop = depth_map[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        dist_gt = float(np.min(center_crop))
        dist_gt = min(max(dist_gt, 0.0), 10.0) # Clamp 0-10m
        
        # Dummy pose to maintain API compatibility
        pose_tensor = torch.zeros(7, dtype=torch.float32)
        return {
            'rgb': rgb_tensor,
            'pose': pose_tensor,
            'depth': depth_tensor,
            'lux': sample['lux'],
            'seq_id': seq_id
        }

class PyBulletDataset(Dataset):
    def __init__(self, dataset_dir="dataset_pybullet", transform=None, target_size=(224, 224)):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_size = target_size
        
        self.labels = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
        self.samples = []
        for idx, row in self.labels.iterrows():
            self.samples.append({
                'frame': int(row['frame']),
                'lux': 10, # Giả lập lux cao (trời sáng) để qua bộ lọc
                'seq_id': 9999 # Dùng ID 9999 để đánh dấu đây là PyBullet data
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        f = sample['frame']
        
        # 1. Load RGB frame
        rgb_path = os.path.join(self.dataset_dir, "rgb", f"{f:04d}.jpg")
        frame_bgr = cv2.imread(rgb_path)
        frame_resized = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            rgb_tensor = self.transform(frame_rgb)
        else:
            rgb_np = frame_rgb.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1) # (3, H, W)
            
        # 2. Load Pose (we just need the pose tensor to be shape (7,) for compatibility, even though we use dist_gt directly later)
        row = self.labels.iloc[idx]
        pose_tensor = torch.tensor([row['x'], row['y'], row['z'], row['qx'], row['qy'], row['qz'], row['qw']], dtype=torch.float32)
        
        # 3. Load Depth
        depth_path = os.path.join(self.dataset_dir, "depth", f"{f:04d}.npy")
        depth_np = np.load(depth_path).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_np)
        
        return {
            'rgb': rgb_tensor,
            'pose': pose_tensor,
            'depth': depth_tensor,
            'lux': sample['lux'],
            'seq_id': sample['seq_id']
        }

def get_weighted_sampler(dataset, low_light_target_ratio=0.22):
    num_low = 0
    num_high = 0
    
    for sample in dataset.samples:
        if sample['lux'] <= 3:
            num_low += 1
        else:
            num_high += 1
            
    if num_low == 0:
        return WeightedRandomSampler(
            weights=[1.0] * len(dataset),
            num_samples=len(dataset),
            replacement=True
        )
        
    w_low = low_light_target_ratio / num_low
    w_high = (1.0 - low_light_target_ratio) / num_high
    
    weights = []
    for sample in dataset.samples:
        if sample['lux'] <= 3:
            weights.append(w_low)
        else:
            weights.append(w_high)
            
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
