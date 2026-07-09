import torch
import jax
import jax.numpy as jnp
from jax import jit
import functools
from torch.utils.dlpack import to_dlpack
from jax.dlpack import from_dlpack

def pytorch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """Zero-copy conversion from PyTorch GPU Tensor to JAX Array."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return from_dlpack(tensor)

@functools.partial(jit, static_argnums=(8,))
def _compute_sparse_pc_jax(
    depth: jnp.ndarray,
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    max_depth: float,
    max_points: int, key: jax.Array
):
    """JAX JIT function to convert depth to a sparse 3D point cloud in Camera Frame."""
    depth_flat = depth.flatten()
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    valid_mask = (depth_flat > 0.1) & (depth_flat < max_depth)
    
    z_cam = depth_flat
    x_cam = (u_flat - cx) * z_cam / fx
    y_cam = (v_flat - cy) * z_cam / fy
    points = jnp.stack([x_cam, y_cam, z_cam], axis=-1)
    
    probs = valid_mask.astype(jnp.float32)
    # If no valid points, probabilities will sum to 0. JAX choice handles this by returning 0 index if p=0
    # but to be safe, we add a tiny epsilon.
    probs = probs + 1e-9
    
    sampled_indices = jax.random.choice(key, jnp.arange(len(depth_flat)), shape=(max_points,), p=probs, replace=True)
    sparse_points = points[sampled_indices]
    
    # Optional: We could mask invalid points to 1e5 here, but since they have depth > max_depth 
    # they will be ignored anyway, or we just rely on valid sampling.
    return sparse_points

@functools.partial(jit, static_argnums=(7,))
def _transform_and_filter_jax(
    local_points: jnp.ndarray, 
    R_w: jnp.ndarray, 
    p_w: jnp.ndarray, 
    pitch_rad: float,
    buffer: jnp.ndarray,
    frame_idx: int,
    voxel_size: float,
    max_unique: int
):
    """
    1. Transforms local Camera points to Global World points.
    2. Inserts into the Rolling Buffer.
    3. Applies Voxel Grid Filter (Rounding) and extracts Unique points statically.
    """
    # --- 1. Camera Frame to Drone Body Frame (Pitch Correction) ---
    cos_p = jnp.cos(pitch_rad)
    sin_p = jnp.sin(pitch_rad)
    
    x_cam = local_points[:, 0]
    y_cam = local_points[:, 1]
    z_cam = local_points[:, 2]
    
    x_drone = x_cam
    y_drone = z_cam * cos_p + y_cam * sin_p
    z_drone = z_cam * sin_p - y_cam * cos_p
    
    body_points = jnp.stack([x_drone, y_drone, z_drone], axis=-1)
    
    # --- 2. Drone Body Frame to Global World Frame ---
    # R_w is (3, 3), body_points is (N, 3). 
    # Global = (R_w @ body_points.T).T + p_w
    global_points = jnp.dot(body_points, R_w.T) + p_w
    
    # --- 3. Update Rolling Buffer ---
    # We use dynamic_update_slice to update the buffer at frame_idx without breaking JIT
    buffer = jax.lax.dynamic_update_slice(
        buffer, 
        jnp.expand_dims(global_points, axis=0), 
        (frame_idx, 0, 0)
    )
    
    # --- 4. Voxel Grid Downsampling (Unique Filter) ---
    # Flatten the buffer from (K, N, 3) to (K*N, 3)
    flat_buffer = buffer.reshape(-1, 3)
    
    # Round coordinates to nearest voxel_size (e.g., 0.1m)
    rounded_points = jnp.round(flat_buffer / voxel_size) * voxel_size
    
    # Extract unique points statically. 
    # Points padded with 1e5 will also be uniquely grouped into a single "garbage" point.
    # size=max_unique forces the output to be exactly (max_unique, 3), padding with fill_value
    unique_points = jnp.unique(rounded_points, axis=0, size=max_unique, fill_value=1e5)
    
    return buffer, unique_points


class RollingSparseMap:
    def __init__(self, h: int, w: int, fx: float, fy: float, cx: float, cy: float, 
                 max_frames: int = 30, points_per_frame: int = 2048, 
                 voxel_size: float = 0.1, max_unique: int = 5000):
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
        self.points_per_frame = points_per_frame
        self.max_frames = max_frames
        self.voxel_size = voxel_size
        self.max_unique = max_unique
        
        self.rng_key = jax.random.PRNGKey(42)
        
        # Pre-compute pixel coordinate grids in JAX
        v_coords = jnp.arange(h)
        u_coords = jnp.arange(w)
        v_grid, u_grid = jnp.meshgrid(v_coords, u_coords, indexing='ij')
        self.u_grid = u_grid
        self.v_grid = v_grid
        
        # The rolling buffer holding global coordinates. Shape: (30, 2048, 3)
        # Initialized with massive numbers (1e5) so they are ignored by collision checks.
        self.buffer = jnp.ones((max_frames, points_per_frame, 3), dtype=jnp.float32) * 1e5
        self.frame_idx = 0
        
        print(f"[RollingSparseMap] JIT Compiling functions (Buffer: {max_frames} frames, {points_per_frame} pts/frame)...")
        # Dummy pass to trigger JIT compilation
        dummy_depth = jnp.ones((h, w), dtype=jnp.float32)
        dummy_pc = _compute_sparse_pc_jax(
            dummy_depth, self.u_grid, self.v_grid,
            self.fx, self.fy, self.cx, self.cy, 5.0, self.points_per_frame, self.rng_key
        )
        dummy_R = jnp.eye(3, dtype=jnp.float32)
        dummy_p = jnp.zeros(3, dtype=jnp.float32)
        _transform_and_filter_jax(
            dummy_pc, dummy_R, dummy_p, 0.0, self.buffer, 0, self.voxel_size, self.max_unique
        )
        print("[RollingSparseMap] Ready!")

    def update(self, depth_tensor: torch.Tensor,
               R_w: jnp.ndarray, p_w: jnp.ndarray, pitch_deg: float, 
               max_depth: float = 5.0) -> jnp.ndarray:
        """
        Updates the map with a new frame and returns the filtered, static-size Global Point Cloud.
        """
        # 1. Zero-copy transfer
        depth_jax = pytorch_to_jax(depth_tensor[0, 0])
        
        # 2. Extract Sparse PC in Camera Frame
        self.rng_key, subkey = jax.random.split(self.rng_key)
        local_points = _compute_sparse_pc_jax(
            depth_jax, self.u_grid, self.v_grid,
            self.fx, self.fy, self.cx, self.cy,
            max_depth, self.points_per_frame, subkey
        )
        
        # 3. Transform to Global, update buffer, and Voxel filter
        pitch_rad = pitch_deg * (jnp.pi / 180.0)
        self.buffer, unique_points = _transform_and_filter_jax(
            local_points, R_w, p_w, pitch_rad, 
            self.buffer, self.frame_idx, self.voxel_size, self.max_unique
        )
        
        # 4. Advance Rolling Buffer Index
        self.frame_idx = (self.frame_idx + 1) % self.max_frames
        
        # Returns exactly (max_unique, 3) array. Garbage points are at (100000.0, 100000.0, 100000.0)
        return unique_points
