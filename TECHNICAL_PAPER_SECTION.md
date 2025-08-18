# Hierarchical State-Based Autonomous Navigation Framework for Vision-Guided UAV Precision Tasks

## Abstract

This paper presents a hierarchical state-based navigation framework for autonomous Unmanned Aerial Vehicles (UAVs) performing precision tasks requiring seamless transitions between GPS-based waypoint navigation and vision-based servoing. The proposed system integrates a finite state machine architecture with adaptive control strategies, incorporating real-time visual fiducial tracking, artificial potential field-based obstacle avoidance, and hybrid position-yaw control mechanisms. The framework addresses critical challenges in autonomous UAV operations including mode transition hysteresis, visual target ambiguity resolution, and robust state management under sensor uncertainty.

## 1. System Architecture and Mathematical Formulation

### 1.1 State Space Definition

The proposed system operates within a discrete state space S defined as:

```
S = {s_idle, s_gps, s_visual, s_search, s_task, s_return, s_land}
```

where each state s_i ∈ S represents a distinct operational mode with associated control policies π_i: X → U, mapping the system state space X ⊂ ℝ^n to the control input space U ⊂ ℝ^m.

The system state vector at time t is defined as:

```
x(t) = [p^T(t), v^T(t), q(t), ω(t), ξ(t)]^T
```

where:
- p(t) = [x, y, z]^T ∈ ℝ^3 represents the position in the world frame
- v(t) = [v_x, v_y, v_z]^T ∈ ℝ^3 denotes the linear velocity
- q(t) = [q_w, q_x, q_y, q_z]^T ∈ SO(3) represents the orientation quaternion
- ω(t) = [ω_x, ω_y, ω_z]^T ∈ ℝ^3 denotes the angular velocity
- ξ(t) ∈ ℝ^k represents auxiliary state variables including ArUco detection confidence

### 1.2 State Transition Model

The state transition function is formulated as a Mealy machine:

```
δ: S × I × C → S
λ: S × I → O
```

where:
- I represents the input alphabet (sensor measurements and detection events)
- C represents the set of guard conditions
- O represents the output alphabet (control mode commands)

The transition from GPS navigation (s_gps) to visual servoing (s_visual) is governed by the guard condition:

```
G_gps→visual = C_aruco ∧ C_range ∧ C_proximity
```

where:
- C_aruco: ArUco detection confidence exceeds threshold
- C_range: ||p_target - p|| ≤ r_visual
- C_proximity: ||p_goal - p|| ≤ r_gps_tolerance

### 1.3 ArUco Detection Confidence Model

The ArUco detection confidence is computed using a sliding window approach with temporal filtering:

```
γ(t) = (1/W) Σ_{i=t-W+1}^{t} I_detected(i)
```

where:
- W is the window size (typically 30 frames)
- I_detected(i) is an indicator function: 1 if ArUco detected at frame i, 0 otherwise

The detection is considered confident when:

```
C_aruco = (γ(t) ≥ θ_confidence)
```

with θ_confidence = 0.7 (70% detection rate within the window).

## 2. Visual Servoing and Pose Estimation

### 2.1 ArUco Pose Estimation

Given a detected ArUco marker with corners c_i ∈ ℝ^2, i ∈ {1,2,3,4} in image coordinates, the marker pose relative to the camera frame is estimated using the Perspective-n-Point (PnP) algorithm:

```
[R_m^c, t_m^c] = PnP(c, K, D, l_marker)
```

where:
- R_m^c ∈ SO(3) is the rotation matrix from marker to camera frame
- t_m^c ∈ ℝ^3 is the translation vector
- K ∈ ℝ^{3×3} is the camera intrinsic matrix
- D ∈ ℝ^5 represents distortion coefficients
- l_marker is the physical marker size

### 2.2 Ambiguity Resolution

The inherent ambiguity in planar marker pose estimation is resolved through geometric consistency checking:

```
φ = n_m^T · t_m^c
```

where n_m = R_m^c[:, 2] is the marker's normal vector. If φ > 0, indicating the marker faces away from the camera, the pose is corrected:

```
R_m^c_corrected = R_m^c · R_y(π)
```

where R_y(π) represents a 180° rotation about the marker's Y-axis.

### 2.3 Goal Pose Computation

The desired UAV pose for the injection task is computed as:

```
T_goal^c = T_m^c · T_offset
```

where T_offset encodes the desired standoff distance and orientation:

```
T_offset = [
    R_offset | t_offset
    0^T      | 1
]
```

with t_offset = [x_off, y_off, z_off]^T representing the injection position offset from the marker center.

The goal orientation is defined to ensure the UAV faces the marker:

```
R_goal^c = [-z_m, x_m, -y_m]
```

where x_m, y_m, z_m are the marker's axis vectors in the camera frame.

## 3. Control Strategy

### 3.1 Hierarchical Control Architecture

The control system employs a hierarchical architecture with three layers:

1. **Strategic Layer**: Finite state machine for mission-level decisions
2. **Tactical Layer**: Mode-specific trajectory generation
3. **Reactive Layer**: Low-level control with obstacle avoidance

### 3.2 Position Control with Trajectory Awareness

The position controller implements a hybrid PID structure with mode-dependent gains:

```
u_pos(t) = K_p^(m) e_p(t) + K_i^(m) ∫e_p(τ)dτ + K_d^(m) ė_p(t)
```

where:
- e_p(t) = p_desired(t) - p(t) is the position error
- K_p^(m), K_i^(m), K_d^(m) are mode-dependent gain matrices
- m ∈ {gps, visual, trajectory} represents the control mode

For trajectory tracking mode, a feedforward term is added:

```
u_traj(t) = u_pos(t) + α_ff v_ref(t)
```

where v_ref(t) is the reference velocity from the trajectory planner and α_ff is the feedforward gain.

### 3.3 Hybrid Yaw Control

The yaw control strategy adapts based on the distance to target:

```
ψ_desired = {
    atan2(v_y, v_x),           if d > d_far
    atan2(Δy, Δx),             if d_near < d ≤ d_far
    ψ_target,                   if d ≤ d_near
}
```

where:
- d = ||p_target - p|| is the distance to target
- d_far = 60m is the far-field threshold
- d_near = 1.5m is the near-field threshold
- Δ = p_target - p is the position error vector

The yaw rate command is computed using proportional control:

```
ω_z = K_ψ · wrap_to_pi(ψ_desired - ψ_current)
```

where wrap_to_pi normalizes the angle difference to [-π, π].

### 3.4 Search Pattern Generation

During the search state (s_search), when visual target is lost, the system executes a spiral search pattern:

```
p_search(t) = p_center + r(t)[cos(θ(t)), sin(θ(t)), 0]^T
```

where:
- r(t) = r_0 + v_r · t is the expanding radius
- θ(t) = ω_search · t is the angular position
- p_center is the last known target position

## 4. Obstacle Avoidance using Artificial Potential Fields

### 4.1 Potential Field Formulation

The artificial potential field combines attractive and repulsive components:

```
U_total(p) = U_att(p) + U_rep(p)
```

The attractive potential is quadratic:

```
U_att(p) = (1/2) k_att ||p - p_goal||^2
```

The repulsive potential follows the FIRAS (Force Inducing an Artificial Repulsion from the Surface) model:

```
U_rep(p) = Σ_i η_i · H(r - d_i) · (1/d_i - 1/r)^2
```

where:
- η_i is the repulsion gain for obstacle i
- d_i = ||p - p_obs,i|| is the distance to obstacle i
- r is the influence radius
- H(·) is the Heaviside step function

### 4.2 Velocity Field Computation

The velocity command is derived from the negative gradient of the potential field:

```
v_cmd = -∇U_total = v_att + v_rep
```

where:

```
v_att = k_att(p_goal - p)
```

```
v_rep = -Σ_i η_i · H(r - d_i) · (1/d_i - 1/r) · (1/d_i^3) · (p - p_obs,i)
```

To prevent excessive repulsion, the repulsive velocity is clamped:

```
v_rep = min(||v_rep||, v_rep,max) · v̂_rep
```

### 4.3 Anisotropic Repulsion

To account for the UAV's different maneuverability in horizontal and vertical planes:

```
η = diag([η_xy, η_xy, η_z])
```

where typically η_z < η_xy to reflect reduced vertical agility.

## 5. Multi-Marker Selection Strategy

### 5.1 Marker Filtering

Valid markers must satisfy geometric constraints:

```
A_min ≤ A_marker ≤ A_max
```

where A_marker is the marker's pixel area, computed as:

```
A_marker = ||(c_1 - c_3) × (c_2 - c_4)|| / 2
```

### 5.2 Optimal Marker Selection

When multiple valid markers are detected, selection follows a context-aware strategy:

```
m* = argmin_i f_cost(m_i)
```

where the cost function is:

```
f_cost(m_i) = {
    ||p_m,i - p_goal||,     if mission goal exists
    -A_i,                   if no mission goal
}
```

This ensures selection of the marker closest to the mission goal when available, or the largest (closest) marker otherwise.

## 6. State Estimation and Temporal Filtering

### 6.1 Pose Estimation Filtering

The UAV pose estimate incorporates an Extended Kalman Filter (EKF) with state vector:

```
x̂ = [p^T, v^T, q^T, b_a^T, b_g^T]^T
```

where b_a and b_g represent accelerometer and gyroscope biases.

The prediction step uses the IMU measurements:

```
x̂_k|k-1 = f(x̂_k-1|k-1, u_k, Δt)
v̂_k|k-1 = v̂_k-1|k-1 + (R(q̂_k-1) · (a_m - b̂_a) - g) · Δt
p̂_k|k-1 = p̂_k-1|k-1 + v̂_k-1|k-1 · Δt + (1/2) · a · Δt^2
```

The update step fuses visual measurements when available:

```
K_k = P_k|k-1 H^T (H P_k|k-1 H^T + R)^{-1}
x̂_k|k = x̂_k|k-1 + K_k(z_k - h(x̂_k|k-1))
```

### 6.2 Velocity Smoothing

Output velocity commands are smoothed using exponential moving average:

```
v_smooth(t) = α · v_cmd(t) + (1 - α) · v_smooth(t-1)
```

with α ∈ [0, 1] controlling the smoothing factor.

## 7. Safety Mechanisms and Fault Tolerance

### 7.1 Transition Hysteresis

To prevent oscillatory mode switching, hysteresis is implemented:

```
T_activate = t_current - t_last_transition > Δt_min
```

where Δt_min = 2.0 seconds is the minimum time between transitions.

### 7.2 Failsafe Behaviors

The system implements multiple failsafe mechanisms:

1. **Visual Lock Loss**: Transition to search pattern after τ_timeout = 2.0 seconds
2. **Search Timeout**: Return to GPS navigation after τ_search = 30.0 seconds
3. **Goal Unreachable**: Hover mode activation if ||e_p|| > e_max for t > t_threshold

### 7.3 State Consistency Verification

State consistency is verified through invariant checking:

```
I(s) = {
    s = s_visual ⟹ (C_aruco ∧ visual_target ≠ ∅)
    s = s_gps ⟹ (gps_target ≠ ∅)
    s = s_task ⟹ (||e_p|| < ε_pos ∧ ||e_ψ|| < ε_yaw)
}
```

## 8. Performance Metrics and Convergence Analysis

### 8.1 Convergence Criteria

Visual servoing convergence is defined by simultaneous satisfaction of:

```
||p - p_goal|| ≤ ε_pos = 0.1m
|ψ - ψ_goal| ≤ ε_yaw = 5°
```

### 8.2 Stability Analysis

The closed-loop system stability is analyzed using Lyapunov theory. For the position control subsystem, the Lyapunov function is:

```
V(e) = (1/2) e^T P e + (1/2) ∫e^T Q e dτ
```

where P and Q are positive definite matrices. The time derivative:

```
V̇ = -e^T (K_p P + P K_p) e - e^T Q e < 0
```

ensures asymptotic stability for properly chosen gains.

### 8.3 Computational Complexity

The system's computational complexity per control cycle is:

```
O(n · W + m · log(m) + k^3)
```

where:
- n is the number of detected ArUco markers
- W is the detection confidence window size
- m is the number of obstacle points
- k is the state dimension for EKF

## 9. Implementation Considerations

### 9.1 Real-time Constraints

The control loop operates at f_control = 20 Hz, requiring:

```
t_sense + t_process + t_control < 1/f_control = 50ms
```

### 9.2 Communication Architecture

The system employs a publish-subscribe pattern with topics:
- High-frequency (20 Hz): pose updates, velocity commands
- Medium-frequency (10 Hz): goal updates, state transitions
- Low-frequency (1 Hz): status reports, diagnostics

### 9.3 Coordinate Frame Transformations

All transformations maintain consistency through the transformation chain:

```
T_world^drone = T_world^odom · T_odom^base · T_base^drone
```

with proper time synchronization using ROS tf2 framework.

## Conclusion

The presented hierarchical state-based navigation framework provides a mathematically rigorous approach to autonomous UAV navigation for precision tasks. The integration of visual servoing with GPS navigation, coupled with robust state management and obstacle avoidance, enables reliable execution of complex missions requiring both long-range navigation and precise terminal guidance. The system's modular architecture and formal state machine design ensure predictable behavior while maintaining flexibility for diverse operational scenarios.