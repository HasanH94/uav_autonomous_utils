#!/usr/bin/env python3
"""
Standalone performance test for the optimized VO filter.
Tests the core algorithms without ROS overhead.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Numba functions from the optimized version
from numba import jit, njit, prange
import numba

# Copy the Numba functions from the optimized filter
@njit(fastmath=True, cache=True, parallel=False)
def compute_violations_vectorized(v_camera, obstacles, safety_margin, time_horizon):
    """
    Vectorized violation checking in camera frame.
    Returns boolean mask of violations.
    """
    n_obs = obstacles.shape[0]
    if n_obs == 0:
        return np.zeros(0, dtype=np.bool_)
    
    violations = np.zeros(n_obs, dtype=np.bool_)
    v_norm = np.linalg.norm(v_camera)
    
    if v_norm < 1e-9:
        return violations
    
    v_hat = v_camera / v_norm
    
    for i in prange(n_obs):
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

@njit(fastmath=True, cache=True)
def rodrigues_rotation(vec, axis, angle):
    """Rodrigues rotation formula - optimized for Numba."""
    k = axis / (np.linalg.norm(axis) + 1e-9)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # v*cos + (k x v)*sin + k*(k·v)*(1-cos)
    kdotv = np.dot(k, vec)
    kcrossv = np.array([
        k[1]*vec[2] - k[2]*vec[1],
        k[2]*vec[0] - k[0]*vec[2],
        k[0]*vec[1] - k[1]*vec[0]
    ])
    
    return vec * cos_a + kcrossv * sin_a + k * kdotv * (1.0 - cos_a)

def compute_violations_python(v_camera, obstacles, safety_margin, time_horizon):
    """Pure Python version for comparison."""
    violations = []
    v_norm = np.linalg.norm(v_camera)
    
    if v_norm < 1e-9:
        return np.array([], dtype=bool)
    
    v_hat = v_camera / v_norm
    
    for i in range(obstacles.shape[0]):
        p = obstacles[i, :3]
        r = obstacles[i, 3] + safety_margin
        d = np.linalg.norm(p)
        
        if d <= r:
            violations.append(True)
            continue
        
        p_hat = p / d
        closing = np.dot(v_camera, p_hat)
        
        if closing > 0:
            ttc = (d - r) / closing
            if ttc <= time_horizon:
                theta = np.arcsin(min(1.0, r / d))
                cos_angle = np.dot(v_hat, p_hat)
                if cos_angle >= np.cos(theta):
                    violations.append(True)
                else:
                    violations.append(False)
            else:
                violations.append(False)
        else:
            violations.append(False)
    
    return np.array(violations)

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

def generate_test_obstacles(n_points=1000, seed=42):
    """Generate realistic obstacle distribution."""
    np.random.seed(seed)
    
    # Generate points in a cone in front of the drone
    obstacles = []
    
    for _ in range(n_points):
        # Random points in front hemisphere
        theta = np.random.uniform(0, 2*np.pi)  # Azimuth
        phi = np.random.uniform(0, np.pi/3)    # Elevation (60 degree cone)
        r = np.random.uniform(0.5, 8.0)        # Distance
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        obstacles.append([x, y, z, 0.05])  # 5cm radius per point
    
    return np.array(obstacles, dtype=np.float32)

def benchmark_function(func, *args, n_runs=100, warmup=5):
    """Benchmark a function with warmup runs."""
    # Warmup runs (for JIT compilation)
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

def main():
    print("=" * 60)
    print("VO Filter Performance Test")
    print("=" * 60)
    
    # Test parameters
    safety_margin = np.float32(1.0)
    time_horizon = np.float32(3.0)
    voxel_size = np.float32(0.2)
    
    # Test different obstacle counts
    obstacle_counts = [10, 50, 100, 500, 1000]
    
    print("\n1. VIOLATION CHECKING PERFORMANCE")
    print("-" * 40)
    
    for n_obs in obstacle_counts:
        obstacles = generate_test_obstacles(n_obs)
        v_camera = np.array([1.0, 0.1, 0.0], dtype=np.float32)  # Moving forward
        
        # Test Numba version
        numba_stats = benchmark_function(
            compute_violations_vectorized,
            v_camera, obstacles, safety_margin, time_horizon,
            n_runs=100
        )
        
        # Test Python version
        python_stats = benchmark_function(
            compute_violations_python,
            v_camera, obstacles, safety_margin, time_horizon,
            n_runs=100
        )
        
        speedup = python_stats['mean'] / numba_stats['mean']
        
        print(f"\n{n_obs} obstacles:")
        print(f"  Numba:  {numba_stats['mean']:.3f}ms ± {numba_stats['std']:.3f}ms ({numba_stats['fps']:.1f} Hz)")
        print(f"  Python: {python_stats['mean']:.3f}ms ± {python_stats['std']:.3f}ms ({python_stats['fps']:.1f} Hz)")
        print(f"  Speedup: {speedup:.1f}x")
    
    print("\n2. VOXEL DOWNSAMPLING PERFORMANCE")
    print("-" * 40)
    
    point_counts = [1000, 5000, 10000, 20000]
    
    for n_pts in point_counts:
        # Generate random point cloud
        points = np.random.randn(n_pts, 3).astype(np.float32) * 5.0
        
        stats = benchmark_function(
            voxel_downsample_numpy,
            points, voxel_size,
            n_runs=50
        )
        
        # Count output points
        output = voxel_downsample_numpy(points, voxel_size)
        reduction = (1 - output.shape[0] / n_pts) * 100
        
        print(f"\n{n_pts} points -> {output.shape[0]} points ({reduction:.1f}% reduction)")
        print(f"  Time: {stats['mean']:.3f}ms ± {stats['std']:.3f}ms ({stats['fps']:.1f} Hz)")
    
    print("\n3. COMPLETE PIPELINE SIMULATION")
    print("-" * 40)
    
    def simulate_complete_pipeline(cloud_points, obstacles):
        """Simulate the complete VO filter pipeline."""
        # 1. Voxel downsample
        filtered = voxel_downsample_numpy(cloud_points, voxel_size)
        
        # 2. Get nearest 16 obstacles
        distances = np.linalg.norm(filtered, axis=1)
        k = min(16, filtered.shape[0])
        if k > 0:
            nearest_idx = np.argpartition(distances, k-1)[:k]
            obs = np.zeros((k, 4), dtype=np.float32)
            obs[:, :3] = filtered[nearest_idx]
            obs[:, 3] = 0.05
        else:
            obs = np.zeros((0, 4), dtype=np.float32)
        
        # 3. Check violations
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        violations = compute_violations_vectorized(v, obs, safety_margin, time_horizon)
        
        # 4. Rotate if needed (simplified)
        if np.any(violations):
            v = v * 0.7  # Simple slowdown
        
        return v
    
    print("\nComplete pipeline with 5000 input points:")
    cloud = np.random.randn(5000, 3).astype(np.float32) * 5.0
    obstacles = generate_test_obstacles(100)
    
    pipeline_stats = benchmark_function(
        simulate_complete_pipeline,
        cloud, obstacles,
        n_runs=100,
        warmup=10
    )
    
    print(f"  Total time: {pipeline_stats['mean']:.3f}ms ± {pipeline_stats['std']:.3f}ms")
    print(f"  Rate: {pipeline_stats['fps']:.1f} Hz")
    
    print("\n4. NUMBA COMPILATION OVERHEAD")
    print("-" * 40)
    
    # Force recompilation by clearing cache
    compute_violations_vectorized.recompile()
    
    # Time first call (includes compilation)
    obstacles = generate_test_obstacles(100)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    start = time.perf_counter()
    _ = compute_violations_vectorized(v, obstacles, safety_margin, time_horizon)
    first_call = (time.perf_counter() - start) * 1000
    
    # Time second call (already compiled)
    start = time.perf_counter()
    _ = compute_violations_vectorized(v, obstacles, safety_margin, time_horizon)
    second_call = (time.perf_counter() - start) * 1000
    
    print(f"  First call (with compilation): {first_call:.3f}ms")
    print(f"  Second call (cached): {second_call:.3f}ms")
    print(f"  Compilation overhead: {first_call - second_call:.3f}ms")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nFor typical scenario (100 obstacles after filtering):")
    print(f"  - Violation checking: ~{numba_stats['mean']:.2f}ms")
    print(f"  - Can process at: ~{numba_stats['fps']:.0f} Hz")
    print(f"  - Numba speedup: ~{speedup:.0f}x over pure Python")
    print(f"\nComplete pipeline (5000 points -> 16 obstacles):")
    print(f"  - Total processing: ~{pipeline_stats['mean']:.2f}ms")
    print(f"  - Maximum rate: ~{pipeline_stats['fps']:.0f} Hz")
    
    # Check if Open3D is available
    try:
        import open3d as o3d
        print("\n✓ Open3D is installed - would provide additional 2-3x speedup for voxel filtering")
    except ImportError:
        print("\n✗ Open3D not installed - using NumPy fallback for voxel filtering")
        print("  Install with: pip3 install open3d")

if __name__ == "__main__":
    main()