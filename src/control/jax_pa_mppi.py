import jax
import jax.numpy as jnp
from jax import jit, vmap
import functools

# Enable float64 for JAX if high precision is required (Optional)
# jax.config.update("jax_enable_x64", True)

# ============================================================================
# PA-MPPI for Edge-Deployed UAV Obstacle Avoidance
# ============================================================================
# Adapted from: "PA-MPPI: Perception-Aware MPPI" (arXiv:2509.14978v3)
#
# Key differences from the original PA-MPPI paper:
# 1. Dynamics: Differentially Flat 4-DoF kinematic model (velocity-command)
#    instead of full 13-state quadrotor dynamics (thrust + body-rates).
#    Rationale: We target edge deployment where the low-level PID controller
#    handles attitude stabilization, so the planner operates in velocity space.
#
# 2. Mapping: Rolling Sparse Point Cloud (JAX-native) instead of ROG-Map
#    occupancy grid. No occupied/free/unknown trichotomy available.
#    Therefore, ray-tracing perception cost is replaced with continuous
#    soft-margin collision cost + repulsive field.
#
# 3. Cost Design Contributions:
#    a) Continuous Soft-margin Collision: exp-based proximity penalty with
#       smooth gradient, replacing binary voxel indicator.
#    b) Heading Cost: velocity-to-goal alignment to bias forward motion
#       (not present in original PA-MPPI).
#    c) Warm-start Decay: attenuate tail of previous solution to prevent
#       ghost momentum / backward flight artifacts.
#    d) Asymmetric Progress: penalizes backward motion 3x harder than
#       it rewards forward motion, breaking symmetry traps.
# ============================================================================


@jit
def kinematic_step(state, control, dt):
    """
    Differentially-Flat 4-DoF Kinematic Model.
    State: [px, py, pz, vx, vy, vz, yaw]  (7-dim)
    Control: [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]  (4-dim)

    Models first-order lag to approximate low-level controller response.
    The time constant tau represents PID tracking delay.
    """
    px, py, pz, vx, vy, vz, yaw = state
    vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = control

    # Low-level controller lag (τ ≈ 0.1s, typical for micro-UAV PID)
    tau = 0.1
    alpha = dt / (tau + dt)
    vx_new = vx + alpha * (vx_cmd - vx)
    vy_new = vy + alpha * (vy_cmd - vy)
    vz_new = vz + alpha * (vz_cmd - vz)

    px_new = px + vx_new * dt
    py_new = py + vy_new * dt
    pz_new = pz + vz_new * dt
    yaw_new = yaw + yaw_rate_cmd * dt

    return jnp.array([px_new, py_new, pz_new, vx_new, vy_new, vz_new, yaw_new])


@jit
def compute_trajectory_cost(states, controls, goal_pos, obs_points,
                            safe_radius, d_init):
    """
    Multi-objective cost for a single trajectory of length K.

    Cost terms are structured following PA-MPPI's philosophy:
    - ℓ_goal:      Progress-based goal cost (adapted from PA-MPPI eq. ℓ_goal)
    - ℓ_collision:  Continuous soft-margin collision (our contribution)
    - ℓ_perception: Camera-to-goal alignment (PA-MPPI ℓ_perception, PoI term)
    - ℓ_heading:    Velocity-to-goal alignment (our contribution)
    - ℓ_progress:   Asymmetric progress reward/penalty (our contribution)
    - ℓ_act:        Action magnitude + smoothness (PA-MPPI ℓ_act)
    - ℓ_vel:        Velocity damping near goal (PA-MPPI ℓ_vel)
    - ℓ_terminal:   Terminal safe set — hovering penalty (PA-MPPI V̄)
    - ℓ_floor:      Altitude safety boundary
    """
    K = states.shape[0]

    # ── Precompute shared quantities ──────────────────────────────────────
    positions = states[:, :3]                                   # (K, 3)
    velocities = states[:, 3:6]                                 # (K, 3)
    dists_to_goal = jnp.linalg.norm(positions - goal_pos, axis=-1)  # (K,)

    # Direction from each state to goal (2D, for yaw-related costs)
    d_goal_vec = goal_pos[:2] - states[:, :2]                   # (K, 2)
    d_goal_norm = jnp.linalg.norm(d_goal_vec, axis=-1) + 1e-6  # (K,)
    d_goal_dir = d_goal_vec / d_goal_norm[:, None]              # (K, 2)

    # ── 1. Goal Cost (PA-MPPI style: progress-based) ──────────────────────
    # ℓ_goal = −c_goal · max(0, d_0 − d_k)
    # Reward trajectories that get closer to goal compared to CURRENT position.
    # This prevents greedy behavior: detours around obstacles are rewarded
    # as long as the terminal position is closer than the start.
    c_goal = 8.0
    c_goal_terminal = 40.0
    progress_to_goal = jnp.maximum(0.0, d_init - dists_to_goal)     # (K,)
    goal_cost = -c_goal * jnp.sum(progress_to_goal)
    # Terminal position weight (PA-MPPI ℓ_{goal,H-1}): heavily weight
    # the final position to encourage long-range planning around obstacles
    goal_terminal_cost = -c_goal_terminal * jnp.maximum(0.0, d_init - dists_to_goal[-1])

    # ── 2. Collision Cost (Our contribution: Continuous Soft-margin) ───────
    # Unlike PA-MPPI's binary indicator 𝟙{G(p)≠0}, we use continuous cost
    # because our Rolling Sparse Point Cloud has no free/unknown distinction.
    # Two-layer defense:
    #   a) Hard collision zone (d < safe_radius): massive exponential penalty
    #   b) Soft influence zone (safe_radius < d < safe_margin): gentle repulsion
    diffs = positions[:, None, :] - obs_points[None, :, :]      # (K, M, 3)
    dists = jnp.linalg.norm(diffs, axis=-1)                     # (K, M)
    min_dists = jnp.min(dists, axis=-1)                         # (K,)

    safe_margin = 0.45  # reduced from 0.6 — allow drone to fly through tight gaps
    c_collision = 80.0   # reduced from 150 — less afraid of approaching obstacles
    # Exponential barrier: rises sharply inside safe_margin
    collision_cost = jnp.sum(jnp.where(
        min_dists < safe_margin,
        c_collision * jnp.exp(-(min_dists - safe_radius) / 0.15),
        0.0
    ))
    # Repulsive potential field: gentle global push away from obstacles
    c_repulsive = 0.8  # reduced from 1.5 — don't push drone away from every obstacle globally
    repulsive_cost = jnp.sum(1.0 / (min_dists + 0.5)) * c_repulsive

    # ── 3. Perception Cost: Camera-to-Goal Alignment (PA-MPPI ℓ_perception, PoI) ─
    # ℓ_PoI = c_PoI · (1 − ⟨x̂_WB, d̂_goal⟩)² · 𝟙{d_goal > c_thresh}
    # Encourages the camera (yaw) to face the goal direction.
    # Deactivated when close to goal (no need to look forward when arriving).
    c_poi = 10.0
    c_thresh = 1.0  # deactivate perception cost when closer than 1m to goal
    cam_dir = jnp.stack([-jnp.sin(states[:, 6]),
                          jnp.cos(states[:, 6])], axis=-1)      # (K, 2)
    cam_goal_dot = jnp.sum(cam_dir * d_goal_dir, axis=-1)      # (K,)
    # Squared penalty (matching paper formulation) with distance gate
    perception_cost = jnp.sum(
        jnp.where(dists_to_goal > c_thresh,
                  c_poi * (1.0 - cam_goal_dot) ** 2,
                  0.0)
    )

    # ── 4. Heading Cost: Velocity-to-Goal Alignment (Our contribution) ─────
    # Not present in original PA-MPPI. Biases commanded velocity direction
    # toward the goal, breaking the symmetry of left-vs-right avoidance
    # decisions. Only active when speed is above threshold to avoid
    # penalizing hover/slow maneuvers.
    c_heading = 3.0
    speed_cmd = jnp.linalg.norm(controls[:, :2], axis=-1)      # (K,)
    vel_dir = controls[:, :2] / (speed_cmd[:, None] + 1e-6)    # (K, 2)
    heading_dot = jnp.sum(vel_dir * d_goal_dir, axis=-1)       # (K,)
    heading_cost = jnp.sum(
        jnp.where(speed_cmd > 0.2, c_heading * (1.0 - heading_dot), 0.0)
    )

    # ── 5. Progress Cost: Asymmetric (Our contribution) ────────────────────
    # Rewards forward progress, penalizes backward motion 3× harder.
    # This asymmetry is critical for breaking the forward/backward local
    # minimum that standard MPPI suffers from when facing head-on obstacles.
    c_progress_fwd = 5.0   # increased from 3 — more eager to push forward through gaps
    c_progress_bwd = 12.0  # backward penalty still ~2.4x forward reward
    step_progress = dists_to_goal[:-1] - dists_to_goal[1:]     # >0 = closer
    progress_cost = jnp.sum(
        jnp.where(step_progress > 0,
                  -c_progress_fwd * step_progress,
                   c_progress_bwd * jnp.abs(step_progress))
    )

    # ── 6. Action Cost (PA-MPPI ℓ_act) ─────────────────────────────────────
    # ℓ_act = ‖u‖²_R + ‖Δu‖²_{R_Δ}
    # Penalizes both magnitude and rate-of-change of control inputs.
    # Diagonal weight matrices R and R_Δ (matching paper structure).
    R = jnp.array([0.01, 0.01, 0.02, 0.15])           # action magnitude weights
    R_delta = jnp.array([0.03, 0.03, 0.05, 0.05])     # action smoothness weights
    action_magnitude_cost = jnp.sum(controls ** 2 * R)
    delta_u = controls[1:] - controls[:-1]
    action_smoothness_cost = jnp.sum(delta_u ** 2 * R_delta)

    # ── 7. Velocity Damping Near Goal (PA-MPPI ℓ_vel) ──────────────────────
    # ℓ_vel = exp(−c_vel · d²) · ‖v‖²
    # Encourages the UAV to slow down as it approaches the goal.
    # The exponential gate makes this cost negligible when far from goal.
    c_vel_damp = 2.0   # lower = wider gate, starts braking from further away
    speed_sq = jnp.sum(velocities ** 2, axis=-1)                # (K,)
    vel_damping_cost = jnp.sum(
        jnp.exp(-c_vel_damp * dists_to_goal ** 2) * speed_sq
    ) * 5.0  # stronger damping to prevent goal overshoot/oscillation

    # ── 8. Terminal Safe Set (PA-MPPI V̄) ──────────────────────────────────
    # V̄ = c_safe · 𝟙{‖v‖ > v̲}
    # Large penalty if terminal velocity exceeds safe hover threshold.
    # Prevents high-speed goal approach / crashes.
    c_safe = 50.0
    v_thresh = 0.8  # m/s — max acceptable terminal speed
    terminal_speed = jnp.linalg.norm(velocities[-1])
    terminal_cost = jnp.where(
        terminal_speed > v_thresh,
        c_safe * (terminal_speed - v_thresh) ** 2,
        0.0
    )

    # ── 9. Floor Safety (Altitude boundary) ────────────────────────────────
    # Not in original PA-MPPI (they operate within bounded ROG-Map).
    # Critical for point-cloud pipeline: prevents floor-diving.
    c_floor = 350.0  # strong but allows borderline low-altitude maneuvers
    z_min = 0.4      # raised from 0.3 — telemetry showed z=0.15m near-crash
    z_max = 2.5
    floor_violations = jnp.sum(
        jnp.where(positions[:, 2] < z_min,
                  c_floor * (z_min - positions[:, 2]) ** 2, 0.0) +
        jnp.where(positions[:, 2] > z_max,
                  c_floor * (positions[:, 2] - z_max) ** 2, 0.0)
    )

    # ── Total Cost ─────────────────────────────────────────────────────────
    total_cost = (
        goal_cost + goal_terminal_cost +   # attract to goal
        collision_cost + repulsive_cost +   # avoid obstacles
        perception_cost +                   # keep camera facing goal
        heading_cost +                      # bias velocity toward goal
        progress_cost +                     # asymmetric forward/backward
        action_magnitude_cost +             # penalize large inputs
        action_smoothness_cost +            # penalize jerky control
        vel_damping_cost +                  # slow down near goal
        terminal_cost +                     # safe terminal state
        floor_violations                    # altitude bounds
    )
    return total_cost


# vmap: vectorize cost computation across N parallel trajectories
vmap_compute_cost = vmap(compute_trajectory_cost,
                         in_axes=(0, 0, None, None, None, None))


@functools.partial(jit, static_argnums=(5, 6))
def mppi_rollout(state_0, U_guess, noise, goal_pos, obs_points,
                 N, K, dt, u_min, u_max, lam):
    """
    JAX-Accelerated MPPI Rollout.

    Rolls out N stochastic trajectories of length K in parallel using
    jax.lax.scan (no Python for-loops). The MPPI formula computes
    exponentially-weighted average of perturbed control sequences.
    """
    # Generate N perturbed control sequences: (N, K, 4)
    U_noisy = jnp.clip(U_guess + noise, u_min, u_max)

    # Forward dynamics rollout via jax.lax.scan
    def scan_fn(state, control):
        next_state = kinematic_step(state, control, dt)
        return next_state, next_state

    def single_rollout(controls):
        _, states = jax.lax.scan(scan_fn, state_0, controls)
        return states

    all_states = vmap(single_rollout)(U_noisy)  # (N, K, 7)

    # Initial distance to goal (for progress-based cost)
    d_init = jnp.linalg.norm(state_0[:3] - goal_pos)

    # Compute cost for all N trajectories
    costs = vmap_compute_cost(all_states, U_noisy, goal_pos, obs_points,
                              0.3, d_init)  # safe_radius=0.3

    # Core MPPI: softmax-weighted average of control sequences
    beta = jnp.min(costs)
    weights = jnp.exp(-(1.0 / lam) * (costs - beta))
    weights = weights / (jnp.sum(weights) + 1e-10)

    U_opt = jnp.sum(weights[:, None, None] * U_noisy, axis=0)

    return U_opt, all_states, costs


class JaxPAMPPIController:
    """
    PA-MPPI Controller adapted for edge-deployed UAV obstacle avoidance.

    Operates in velocity-command space (4-DoF: vx, vy, vz, yaw_rate)
    with a kinematic prediction model. Designed for real-time execution
    on embedded GPU via JAX JIT compilation.

    Parameters match PA-MPPI paper Table I where applicable:
    - N: number of samples (10,000 — paper uses 17,500)
    - K: horizon steps (30 — paper uses 15 with dual timestep)
    - λ: temperature (paper uses 0.02, we use 1.0 for velocity-space)
    - dt: prediction timestep (0.05s — paper uses Δt_pred=0.1s)
    """
    def __init__(self, horizon=30, num_samples=10000, dt=0.05):
        self.K = horizon
        self.N = num_samples
        self.dt = dt
        self.lam = 1.0  # Temperature (higher for velocity-space)

        # Control bounds: [vx, vy, vz, yaw_rate]
        self.u_min = jnp.array([-1.5, -1.5, -0.5, -1.5])
        self.u_max = jnp.array([ 1.5,  1.5,  0.5,  1.5])
        self.noise_sigma = jnp.array([0.5, 0.5, 0.1, 0.3])

        self.U_guess = jnp.zeros((self.K, 4))
        self.rng_key = jax.random.PRNGKey(42)

        print(f"[PA-MPPI] Initializing JAX PA-MPPI: N={self.N}, K={self.K}, "
              f"dt={self.dt}, λ={self.lam}. Compiling JIT...")

        # Warm-up JIT compilation during init
        dummy_state = jnp.zeros(7)
        dummy_noise = jnp.zeros((self.N, self.K, 4))
        dummy_goal = jnp.array([10.0, 10.0, 1.0])
        dummy_obs = jnp.zeros((10, 3))

        _ = mppi_rollout(
            dummy_state, self.U_guess, dummy_noise, dummy_goal, dummy_obs,
            self.N, self.K, self.dt, self.u_min, self.u_max, self.lam
        )
        print("[PA-MPPI] JIT Compilation successful! Ready to fly.")

    def compute_action(self, state_0, goal_pos, obs_points):
        """
        Compute optimal action for current state.

        Args:
            state_0: [px, py, pz, vx, vy, vz, yaw] — JAX array (7,)
            goal_pos: [gx, gy, gz] — JAX array (3,)
            obs_points: (M, 3) — obstacle points from RollingSparseMap

        Returns:
            action: [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd] — (4,)
            best_traj: (K, 7) — best trajectory states
            all_states: (N, K, 7) — all sampled trajectories
            costs: (N,) — cost of each trajectory
        """
        # Emergency recovery: if drone is falling, command gentle ascent
        if state_0[2] < 0.4:
            self.U_guess = jnp.zeros((self.K, 4))
            self.U_guess = self.U_guess.at[:, 2].set(0.3)

        # 1. Generate noise
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.normal(subkey, shape=(self.N, self.K, 4)) * self.noise_sigma

        # 2. Run MPPI optimization
        self.U_guess, all_states, costs = mppi_rollout(
            jnp.array(state_0), self.U_guess, noise,
            jnp.array(goal_pos), obs_points,
            self.N, self.K, self.dt, self.u_min, self.u_max, self.lam
        )

        # 3. Extract best trajectory
        best_idx = jnp.argmin(costs)
        best_traj = all_states[best_idx]

        # 4. Warm-start with Decay for next cycle
        # Shift sequence forward by 1 step, decay the tail to prevent
        # ghost momentum (our contribution — not in original PA-MPPI)
        action = self.U_guess[0]
        self.U_guess = jnp.roll(self.U_guess, shift=-1, axis=0)
        self.U_guess = self.U_guess.at[-1].set(self.U_guess[-2] * 0.5)

        return action, best_traj, all_states, costs
