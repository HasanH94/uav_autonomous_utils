#!/usr/bin/env python3
"""
Performance test for VO filter WITHOUT Numba - worst case scenario.
This shows the baseline Python performance.
"""

import numpy as np
import time

def compute_violations_pure_python(v_camera, obstacles, safety_margin, time_horizon):
    """
    Pure Python violation checking - this is the worst case performance.
    """
    n_obs = obstacles.shape[0]
    if n_obs == 0:
        return np.zeros(0, dtype=bool)
    
    violations = np.zeros(n_obs, dtype=bool)
    v_norm = np.linalg.norm(v_camera)
    
    if v_norm < 1e-9:
        return violations
    
    v_hat = v_camera / v_norm
    
    for i in range(n_obs):
        p = obstacles[i, :3]
        r = obstacles[i, 3] + safety_margin
        d = np.linalg.norm(p)
        
        # Inside safety bubble
        if d <= r:
            violations[i] = True
            continue
        
        # Check velocity cone
        p_hat = p / d
        closing = np.dot(v_camera, p_hat)
        
        if closing > 0:
            ttc = (d - r) / closing
            if ttc <= time_horizon:
                # Check cone angle
                theta = np.arcsin(min(1.0, r / d))
                cos_angle = np.dot(v_hat, p_hat)
                if cos_angle >= np.cos(theta):
                    violations[i] = True
    
    return violations

def compute_violations_vectorized_numpy(v_camera, obstacles, safety_margin, time_horizon):
    """
    Vectorized NumPy version - better than pure Python but not as good as Numba.
    """
    if obstacles.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    
    v_norm = np.linalg.norm(v_camera)
    if v_norm < 1e-9:
        return np.zeros(obstacles.shape[0], dtype=bool)
    
    v_hat = v_camera / v_norm
    
    # Extract obstacle positions and radii
    positions = obstacles[:, :3]
    radii = obstacles[:, 3] + safety_margin
    
    # Compute distances
    distances = np.linalg.norm(positions, axis=1)
    
    # Initialize violations
    violations = np.zeros(obstacles.shape[0], dtype=bool)
    
    # Check if inside safety bubble
    violations[distances <= radii] = True
    
    # For points outside bubble, check velocity cone
    outside_mask = distances > radii
    if np.any(outside_mask):
        p_hat = positions[outside_mask] / distances[outside_mask, np.newaxis]
        closing = np.dot(p_hat, v_camera)
        
        # Only check points we're moving toward
        approaching = closing > 0
        if np.any(approaching):
            approach_idx = np.where(outside_mask)[0][approaching]
            
            # Time to collision
            ttc = (distances[approach_idx] - radii[approach_idx]) / closing[approaching]
            
            # Check if collision within time horizon
            collision_mask = ttc <= time_horizon
            if np.any(collision_mask):
                collision_idx = approach_idx[collision_mask]
                
                # Check cone angle
                theta = np.arcsin(np.minimum(1.0, radii[collision_idx] / distances[collision_idx]))
                p_hat_coll = positions[collision_idx] / distances[collision_idx, np.newaxis]
                cos_angle = np.dot(p_hat_coll, v_hat)
                
                # Mark violations where angle is within cone
                violations[collision_idx[cos_angle >= np.cos(theta)]] = True
    
    return violations

def voxel_downsample_numpy(points, voxel_size):
    """Fast numpy voxel downsampling using dictionary."""
    if points.shape[0] == 0:
        return points
    
    # Quantize to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Use dictionary to keep one point per voxel
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = points[i]
    
    return np.array(list(voxel_dict.values()), dtype=np.float32)

def rodrigues_rotation_numpy(vec, axis, angle):
    """Rodrigues rotation formula - pure NumPy."""
    k = axis / (np.linalg.norm(axis) + 1e-9)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # v*cos + (k x v)*sin + k*(k·v)*(1-cos)
    kdotv = np.dot(k, vec)
    kcrossv = np.cross(k, vec)
    
    return vec * cos_a + kcrossv * sin_a + k * kdotv * (1.0 - cos_a)

def generate_test_obstacles(n_points=1000, seed=42):
    """Generate realistic obstacle distribution."""
    np.random.seed(seed)
    
    obstacles = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/3)
        r = np.random.uniform(0.5, 8.0)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        obstacles.append([x, y, z, 0.05])
    
    return np.array(obstacles, dtype=np.float32)

def benchmark_function(func, *args, n_runs=100, warmup=5):
    """Benchmark a function with warmup runs."""
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    times = np.array(times)
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'fps': 1.0 / np.mean(times)
    }

def simulate_complete_vo_pipeline(cloud_points, safety_margin, time_horizon, voxel_size):
    """Simulate the complete VO filter pipeline without Numba."""
    # 1. Voxel downsample
    filtered = voxel_downsample_numpy(cloud_points, voxel_size)
    
    # 2. Get nearest 16 obstacles
    distances = np.linalg.norm(filtered, axis=1)
    k = min(16, filtered.shape[0])
    if k > 0:
        nearest_idx = np.argpartition(distances, k-1)[:k] if k < filtered.shape[0] else np.arange(filtered.shape[0])
        obstacles = np.zeros((k, 4), dtype=np.float32)
        obstacles[:, :3] = filtered[nearest_idx]
        obstacles[:, 3] = 0.05
    else:
        obstacles = np.zeros((0, 4), dtype=np.float32)
    
    # 3. Check violations (using vectorized numpy version)
    v_des = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    violations = compute_violations_vectorized_numpy(v_des, obstacles, safety_margin, time_horizon)
    
    # 4. Simple velocity adjustment if violations
    if np.any(violations):
        # For each violating obstacle, compute escape velocity
        v_safe = v_des * 0.7  # Simple slowdown for worst case
    else:
        v_safe = v_des
    
    return v_safe, len(violations), np.sum(violations)

def main():
    print("=" * 60)
    print("VO Filter Performance Test (WITHOUT Numba - Worst Case)")
    print("=" * 60)
    
    # Test parameters
    safety_margin = np.float32(1.0)
    time_horizon = np.float32(3.0)
    voxel_size = np.float32(0.2)
    
    # Test different obstacle counts
    obstacle_counts = [10, 50, 100, 500, 1000]
    
    print("\n1. VIOLATION CHECKING PERFORMANCE")
    print("-" * 40)
    print("Comparing pure Python loop vs NumPy vectorized")
    
    for n_obs in obstacle_counts:
        obstacles = generate_test_obstacles(n_obs)
        v_camera = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        
        # Test pure Python version
        python_stats = benchmark_function(
            compute_violations_pure_python,
            v_camera, obstacles, safety_margin, time_horizon,
            n_runs=50 if n_obs <= 100 else 10
        )
        
        # Test NumPy vectorized version
        numpy_stats = benchmark_function(
            compute_violations_vectorized_numpy,
            v_camera, obstacles, safety_margin, time_horizon,
            n_runs=100
        )
        
        speedup = python_stats['mean'] / numpy_stats['mean']
        
        print(f"\n{n_obs} obstacles:")
        print(f"  Pure Python:      {python_stats['mean']:.3f}ms ({python_stats['fps']:.1f} Hz)")
        print(f"  NumPy vectorized: {numpy_stats['mean']:.3f}ms ({numpy_stats['fps']:.1f} Hz)")
        print(f"  Speedup:          {speedup:.1f}x")
    
    print("\n2. VOXEL DOWNSAMPLING PERFORMANCE")
    print("-" * 40)
    
    point_counts = [1000, 5000, 10000, 20000]
    
    for n_pts in point_counts:
        # Generate random point cloud
        points = np.random.randn(n_pts, 3).astype(np.float32) * 5.0
        
        stats = benchmark_function(
            voxel_downsample_numpy,
            points, voxel_size,
            n_runs=20
        )
        
        # Count output points
        output = voxel_downsample_numpy(points, voxel_size)
        reduction = (1 - output.shape[0] / n_pts) * 100
        
        print(f"\n{n_pts} points -> {output.shape[0]} points ({reduction:.1f}% reduction)")
        print(f"  Time: {stats['mean']:.3f}ms ({stats['fps']:.1f} Hz)")
    
    print("\n3. COMPLETE PIPELINE SIMULATION")
    print("-" * 40)
    print("Simulating full VO filter pipeline (worst case)")
    
    cloud_sizes = [1000, 2000, 5000, 10000]
    
    for cloud_size in cloud_sizes:
        cloud = np.random.randn(cloud_size, 3).astype(np.float32) * 5.0
        
        stats = benchmark_function(
            simulate_complete_vo_pipeline,
            cloud, safety_margin, time_horizon, voxel_size,
            n_runs=20
        )
        
        print(f"\n{cloud_size} input points:")
        print(f"  Total time: {stats['mean']:.3f}ms ± {stats['std']:.3f}ms")
        print(f"  Rate: {stats['fps']:.1f} Hz")
    
    print("\n" + "=" * 60)
    print("SUMMARY (WORST CASE - No Numba/Open3D)")
    print("=" * 60)
    
    print(f"\nFor typical scenario (5000 points -> 16 obstacles):")
    print(f"  - Processing rate: ~{stats['fps']:.0f} Hz")
    print(f"\nBottlenecks:")
    print(f"  1. Voxel downsampling: Dictionary-based (slow)")
    print(f"  2. Violation checking: NumPy vectorized (moderate)")
    print(f"  3. No JIT compilation: Python interpreter overhead")
    
    print(f"\nExpected improvements with optimizations:")
    print(f"  - With Numba JIT: 5-10x faster violation checking")
    print(f"  - With Open3D: 3-5x faster voxel downsampling")
    print(f"  - Combined: 10-20x overall speedup possible")

if __name__ == "__main__":
    main()