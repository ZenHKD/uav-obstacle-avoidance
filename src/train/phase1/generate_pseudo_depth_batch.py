import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import pipeline
import glob
from PIL import Image

def main(args):
    # Setup device
    device = 0 if torch.cuda.is_available() else -1
    
    # =========================================================================
    # Performance Optimizations for Ampere+ GPUs (e.g. RTX 3060)
    # =========================================================================
    if device == 0:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Enabled TF32 and cuDNN benchmark for maximum speed.")

    # Load the model using float16 to double the inference speed and halve VRAM usage
    # (Note: Using float16 instead of bfloat16 because NumPy doesn't natively support bfloat16 conversion)
    dtype = torch.float16 if device == 0 else torch.float32
    print(f"Loading Depth Anything Model on device: {'GPU (CUDA)' if device == 0 else 'CPU'} with precision: {dtype}...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device, dtype=dtype)
    # =========================================================================
    
    # Discover all trial directories
    dataset_dir = args.dataset_dir
    trial_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()], key=lambda x: int(x))
    
    print(f"Found {len(trial_dirs)} trial directories in {dataset_dir}.")
    
    if args.dry_run:
        trial_dirs = trial_dirs[:3] # Process only 3 trials for testing
        print("DRY RUN MODE: Only processing first 3 trials.")
        
    for trial_id in tqdm(trial_dirs, desc="Processing Trials (Videos)"):
        trial_path = os.path.join(dataset_dir, trial_id)
        video_path = os.path.join(trial_path, f"{trial_id}.avi")
        
        if not os.path.exists(video_path):
            continue
            
        # Create output directory for pseudo-depth
        out_dir = os.path.join(trial_path, "pseudo_depth")
        os.makedirs(out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            continue
            
        # Skip logic to allow resuming if process is interrupted
        if args.skip_existing:
            existing_files = glob.glob(os.path.join(out_dir, "*.npy"))
            # If 95% of frames are processed, we assume it's done (cv2 frame count can be slightly off)
            if len(existing_files) >= total_frames * 0.95:
                cap.release()
                continue
        
        # Process in batches for faster inference
        batch_size = 64
        frame_buffer = []
        frame_idx_buffer = []
        
        for frame_idx in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            out_file = os.path.join(out_dir, f"frame_{frame_idx:04d}.npy")
            if args.skip_existing and os.path.exists(out_file):
                continue
                
            # Resize input directly to 224x224. This saves inference time, matches the student's 
            # expected aspect ratio (squished), and automatically forces the output to be 224x224!
            frame_resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            frame_buffer.append(pil_image)
            frame_idx_buffer.append(frame_idx)
            
            # When buffer is full or at the end of video, run batched inference
            if len(frame_buffer) == batch_size or frame_idx == total_frames - 1:
                # Run batched inference
                results = pipe(frame_buffer, batch_size=len(frame_buffer))
                
                # If batch_size == 1, HF pipeline returns a dict instead of a list of dicts
                if not isinstance(results, list):
                    results = [results]
                    
                for idx, res in zip(frame_idx_buffer, results):
                    depth_image = res["depth"] # Automatically 224x224 because input was 224x224
                    depth_np = np.array(depth_image).astype(np.float16)
                    out_path = os.path.join(out_dir, f"frame_{idx:04d}.npy")
                    np.save(out_path, depth_np)
                    
                # Clear buffers
                frame_buffer = []
                frame_idx_buffer = []
                
        cap.release()
        
    print("\nPhase 1: Pseudo-Depth Generation Completed!")
    print(f"Data saved inside {dataset_dir}/*/pseudo_depth/*.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Batch generate pseudo-depth maps using Depth Anything.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to the root ODA dataset directory.")
    parser.add_argument("--dry_run", action="store_true", help="Run on just 3 trials to verify the pipeline works.")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip frames that already have .npy files (Resume mode).")
    
    args = parser.parse_args()
    main(args)
