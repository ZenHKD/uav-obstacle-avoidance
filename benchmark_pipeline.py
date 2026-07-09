import os
import sys
import time
import numpy as np
import torch

# Ensure JAX does not preallocate 75% of VRAM so we can measure actual usage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
import onnxruntime as ort

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.geometry.jax_sparse_mapping import RollingSparseMap
from src.control.jax_pa_mppi import JaxPAMPPIController

def get_gpu_power_and_vram(handle):
    if not HAS_NVML: return 0.0, 0.0
    try:
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2) # MB
        return power, mem
    except Exception:
        return 0.0, 0.0

def benchmark_loop(mode="GT", num_warmup=20, num_iters=100):
    print(f"\n==========================================")
    print(f"Benchmarking: {mode}")
    print(f"==========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ort_session = None
    
    if mode == "FastViT-DPT":
        onnx_path = "checkpoint/phase4/odanet_fast.onnx"
        providers = ['CUDAExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
    elif mode == "DepthAnythingV2":
        onnx_path = "checkpoint/phase4/depth_anything_v2.onnx"
        providers = ['CUDAExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
    # Init JAX modules
    fx = fy = 163.0
    cx = cy = 112.0
    sparse_map = RollingSparseMap(h=224, w=224, fx=fx, fy=fy, cx=cx, cy=cy, 
                                  max_frames=100, points_per_frame=2048, 
                                  voxel_size=0.15, max_unique=10000)
    controller = JaxPAMPPIController(horizon=30, num_samples=10000, dt=0.05)
    
    # Dummy data
    rgb_input = np.random.rand(224, 224, 3).astype(np.float32)
    depth_gt = np.random.rand(224, 224).astype(np.float32) * 5.0
    
    # Pre-compile JAX
    print("Compiling JAX modules...")
    rot_yaw_only = Rotation.from_euler('xyz', [0.0, 0.0, 0.0]).as_matrix()
    R_w_jax = jnp.array(rot_yaw_only, dtype=jnp.float32)
    p_w_jax = jnp.array([0, 0, 1.0], dtype=jnp.float32)
    goal_pos_jax = jnp.array([0.0, 40.0, 1.0], dtype=jnp.float32)
    state_0_jax = jnp.array([0,0,1, 2,0,0, 0], dtype=jnp.float32)
    
    # Compile step
    dummy_depth_tensor = torch.zeros(1, 1, 224, 224, device=device)
    unique_points_jax = sparse_map.update(dummy_depth_tensor, R_w_jax, p_w_jax, pitch_deg=0.0, max_depth=6.0)
    obs_points_jax = jnp.where(unique_points_jax[:, 2:3] <= 0.15, 1e5, unique_points_jax)
    action, _, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, obs_points_jax)
    action.block_until_ready()
    print("Compilation done.")

    if HAS_NVML:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    else:
        handle = None

    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(num_warmup):
        if mode == "FastViT-DPT":
            rgb_tensor_np = np.expand_dims(np.transpose(rgb_input, (2, 0, 1)), axis=0)
            ort_inputs = {ort_session.get_inputs()[0].name: rgb_tensor_np}
            ort_outs = ort_session.run(None, ort_inputs)
            depth_tensor = torch.from_numpy(ort_outs[0]).to(device)
        elif mode == "DepthAnythingV2":
            rgb_tensor_np = np.expand_dims(np.transpose(rgb_input, (2, 0, 1)), axis=0)
            ort_inputs = {ort_session.get_inputs()[0].name: rgb_tensor_np}
            ort_outs = ort_session.run(None, ort_inputs)
            out_arr = ort_outs[0]
            if len(out_arr.shape) == 3:
                out_arr = np.expand_dims(out_arr, axis=1) # [1, 1, 224, 224]
            depth_tensor = torch.from_numpy(out_arr).to(device)
        else:
            depth_tensor = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).to(device)

        unique_points_jax = sparse_map.update(depth_tensor, R_w_jax, p_w_jax, pitch_deg=0.0, max_depth=6.0)
        obs_points_jax = jnp.where(unique_points_jax[:, 2:3] <= 0.15, 1e5, unique_points_jax)
        action, _, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, obs_points_jax)
        action.block_until_ready()

    # Benchmark
    times = []
    depth_times = []
    mapping_times = []
    mppi_times = []
    powers = []
    vrams = []
    
    # Pre-load GT depth tensor to GPU outside the loop so we don't measure memory transfer
    if mode == "GT":
        depth_tensor = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).to(device)
        
    for _ in range(num_iters):
        t0 = time.perf_counter()
        
        # 1. Depth
        if mode == "FastViT-DPT":
            rgb_tensor_np = np.expand_dims(np.transpose(rgb_input, (2, 0, 1)), axis=0)
            ort_inputs = {ort_session.get_inputs()[0].name: rgb_tensor_np}
            ort_outs = ort_session.run(None, ort_inputs)
            depth_tensor = torch.from_numpy(ort_outs[0]).to(device)
        elif mode == "DepthAnythingV2":
            rgb_tensor_np = np.expand_dims(np.transpose(rgb_input, (2, 0, 1)), axis=0)
            ort_inputs = {ort_session.get_inputs()[0].name: rgb_tensor_np}
            ort_outs = ort_session.run(None, ort_inputs)
            out_arr = ort_outs[0]
            if len(out_arr.shape) == 3:
                out_arr = np.expand_dims(out_arr, axis=1) # [1, 1, 224, 224]
            depth_tensor = torch.from_numpy(out_arr).to(device)
            
        # Synchronize torch if needed (ONNX might be sync, but good practice)
        if torch.cuda.is_available() and mode != "GT":
            torch.cuda.synchronize()
            
        t1 = time.perf_counter()
            
        # 2. Control (Mapping + MPPI)
        unique_points_jax = sparse_map.update(depth_tensor, R_w_jax, p_w_jax, pitch_deg=0.0, max_depth=6.0)
        obs_points_jax = jnp.where(unique_points_jax[:, 2:3] <= 0.15, 1e5, unique_points_jax)
        obs_points_jax.block_until_ready()
        
        t_map = time.perf_counter()
        
        action, _, _, _ = controller.compute_action(state_0_jax, goal_pos_jax, obs_points_jax)
        action.block_until_ready()
        
        t2 = time.perf_counter()
        
        d_time = 0.0 if mode == "GT" else (t1 - t0) * 1000.0
        m_time = (t_map - t1) * 1000.0
        c_time = (t2 - t_map) * 1000.0
        # Recompute l_time as sum to avoid discrepancy when GT d_time is forced to 0
        l_time = d_time + m_time + c_time
        
        depth_times.append(d_time)
        mapping_times.append(m_time)
        mppi_times.append(c_time)
        times.append(l_time)
        
        if HAS_NVML:
            pw, vr = get_gpu_power_and_vram(handle)
            powers.append(pw)
            vrams.append(vr)

    mean_time = np.mean(times)
    std_time = np.std(times)
    
    freqs = 1000.0 / np.array(times)
    mean_freq = np.mean(freqs)
    std_freq = np.std(freqs)
    
    mean_depth_time = np.mean(depth_times)
    std_depth_time = np.std(depth_times)
    mean_mapping_time = np.mean(mapping_times)
    std_mapping_time = np.std(mapping_times)
    mean_mppi_time = np.mean(mppi_times)
    std_mppi_time = np.std(mppi_times)
    
    print(f"Results for {mode}:")
    print(f"  Avg Loop Time  : {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Total Loop Freq: {mean_freq:.1f} ± {std_freq:.1f} Hz")
    print(f"  --> Depth Time : {mean_depth_time:.2f} ± {std_depth_time:.2f} ms")
    print(f"  --> Map Time   : {mean_mapping_time:.2f} ± {std_mapping_time:.2f} ms")
    print(f"  --> MPPI Time  : {mean_mppi_time:.2f} ± {std_mppi_time:.2f} ms")
    
    res_dict = {
        "Model": mode,
        "Time_ms": mean_time,
        "Time_std_ms": std_time,
        "Freq_Hz": mean_freq,
        "Freq_std_Hz": std_freq,
        "Depth_Time_ms": mean_depth_time,
        "Depth_Time_std_ms": std_depth_time,
        "Mapping_Time_ms": mean_mapping_time,
        "Mapping_Time_std_ms": std_mapping_time,
        "MPPI_Time_ms": mean_mppi_time,
        "MPPI_Time_std_ms": std_mppi_time,
        "VRAM_MB": -1,
        "Power_W": -1,
        "Energy_J": -1
    }
    
    if HAS_NVML and len(powers) > 0:
        mean_power = np.mean(powers)
        std_power = np.std(powers)
        
        mean_vram = np.mean(vrams)
        std_vram = np.std(vrams)
        max_vram = np.max(vrams)
        
        # Energy = Power * Time (per loop array)
        energies = np.array(powers) * (np.array(times) / 1000.0)
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        res_dict["VRAM_MB"] = mean_vram
        res_dict["VRAM_std_MB"] = std_vram
        res_dict["Power_W"] = mean_power
        res_dict["Power_std_W"] = std_power
        res_dict["Energy_J"] = mean_energy
        res_dict["Energy_std_J"] = std_energy
        
        print(f"  Total Peak VRAM : {max_vram:.1f} MB (Mean: {mean_vram:.1f} ± {std_vram:.1f} MB)")
        print(f"  Avg GPU Power   : {mean_power:.1f} ± {std_power:.1f} W")
        print(f"  Energy per Loop : {mean_energy:.4f} ± {std_energy:.4f} Joules")
    else:
        res_dict["VRAM_std_MB"] = -1
        res_dict["Power_std_W"] = -1
        res_dict["Energy_std_J"] = -1
        print("  NVML not available. VRAM/Power metrics skipped.")
        
    return res_dict

def save_results(results):
    import csv
    csv_file = "benchmark_metrics.csv"
    txt_file = "benchmark_report.txt"
    
    # Save to CSV
    fields = ["Model", "Time_ms", "Time_std_ms", "Freq_Hz", "Freq_std_Hz", "Depth_Time_ms", "Depth_Time_std_ms", "Mapping_Time_ms", "Mapping_Time_std_ms", "MPPI_Time_ms", "MPPI_Time_std_ms", "VRAM_MB", "VRAM_std_MB", "Power_W", "Power_std_W", "Energy_J", "Energy_std_J"]
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Save to TXT
    with open(txt_file, mode='w') as f:
        f.write("=== JAX UAV OBSTACLE AVOIDANCE BENCHMARK REPORT ===\n\n")
        for r in results:
            f.write(f"Model: {r['Model']}\n")
            f.write(f"  - Avg Loop Time  : {r['Time_ms']:.2f} ± {r['Time_std_ms']:.2f} ms\n")
            f.write(f"  - Total Loop Freq: {r['Freq_Hz']:.1f} ± {r['Freq_std_Hz']:.1f} Hz\n")
            f.write(f"  - Depth Time     : {r['Depth_Time_ms']:.2f} ± {r['Depth_Time_std_ms']:.2f} ms\n")
            f.write(f"  - Mapping Time   : {r['Mapping_Time_ms']:.2f} ± {r['Mapping_Time_std_ms']:.2f} ms\n")
            f.write(f"  - MPPI Time      : {r['MPPI_Time_ms']:.2f} ± {r['MPPI_Time_std_ms']:.2f} ms\n")
            
            if r['VRAM_MB'] != -1:
                f.write(f"  - Avg VRAM       : {r['VRAM_MB']:.1f} ± {r['VRAM_std_MB']:.1f} MB\n")
                f.write(f"  - Avg Power      : {r['Power_W']:.1f} ± {r['Power_std_W']:.1f} W\n")
                f.write(f"  - Energy/Loop    : {r['Energy_J']:.4f} ± {r['Energy_std_J']:.4f} Joules\n")
            f.write("\n")
            
    print(f"\n=> Results saved to {csv_file} and {txt_file}")

if __name__ == "__main__":
    print("Warming up system...")
    time.sleep(2) # settle GPU
    results = []
    results.append(benchmark_loop(mode="GT", num_warmup=20, num_iters=100))
    results.append(benchmark_loop(mode="FastViT-DPT", num_warmup=20, num_iters=100))
    results.append(benchmark_loop(mode="DepthAnythingV2", num_warmup=20, num_iters=100))
    save_results(results)
    print("\nBenchmark completed!")
