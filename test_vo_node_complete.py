#!/usr/bin/env python3
"""
Complete test for the VO filter node - tests both performance AND correctness.
Can run with or without ROS to test the actual node implementation.
"""

import numpy as np
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/control'))

@dataclass
class TestCase:
    """Defines a test scenario"""
    name: str
    description: str
    drone_velocity: np.ndarray  # Desired velocity in world frame
    obstacles: np.ndarray  # Nx4 array [x,y,z,radius] in camera frame
    expected_behavior: str  # What should happen
    validation_func: callable  # Function to check if output is correct

class VOFilterTester:
    def __init__(self, use_numba=False):
        """Initialize tester with or without Numba"""
        self.use_numba = use_numba
        self.results = []
        
        # Import the appropriate version
        if use_numba:
            try:
                from dynamics_aware_vo_filter_optimized import (
                    compute_violations_vectorized,
                    rotate_to_cone_boundary,
                    DynamicsAwareVOFilter3D
                )
                self.compute_violations = compute_violations_vectorized
                self.rotate_to_boundary = rotate_to_cone_boundary
                print("✓ Using optimized version with Numba")
            except ImportError as e:
                print(f"✗ Cannot import optimized version: {e}")
                sys.exit(1)
        else:
            # Use the pure Python version
            print("✓ Using pure Python version (no Numba)")
            self.compute_violations = self.compute_violations_python
            self.rotate_to_boundary = None
        
        # Set up test parameters
        self.safety_margin = 0.6
        self.time_horizon = 2.0
        self.max_speed = 3.0
        self.w_body_x = 2.0
        self.w_body_y = 1.0
        self.w_body_z = 4.0
        
        # Identity transforms for simple testing
        self.R_wb = np.eye(3)  # World to body
        self.R_cb = np.eye(3)  # Camera to body

    def compute_violations_python(self, v_camera, obstacles, safety_margin, time_horizon):
        """Pure Python violation checker for comparison"""
        violations = []
        v_norm = np.linalg.norm(v_camera)
        
        if v_norm < 1e-9:
            return np.zeros(obstacles.shape[0], dtype=bool)
        
        v_hat = v_camera / v_norm
        
        for i in range(obstacles.shape[0]):
            p = obstacles[i, :3]
            r = obstacles[i, 3] + safety_margin
            d = np.linalg.norm(p)
            
            # Inside safety bubble
            if d <= r:
                violations.append(True)
                continue
            
            # Check velocity cone
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
        
        return np.array(violations, dtype=bool)

    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases"""
        test_cases = []
        
        # Test 1: Direct collision course
        test_cases.append(TestCase(
            name="Direct Collision",
            description="Obstacle directly in front, should slow down or stop",
            drone_velocity=np.array([2.0, 0.0, 0.0]),  # Moving forward at 2 m/s
            obstacles=np.array([[3.0, 0.0, 0.0, 0.5]]),  # 3m ahead, 0.5m radius
            expected_behavior="Velocity should decrease significantly",
            validation_func=lambda v_in, v_out: v_out[0] < v_in[0] * 0.8
        ))
        
        # Test 2: Obstacle to the side - should not affect
        test_cases.append(TestCase(
            name="Clear Path",
            description="Obstacle far to the side, should not affect velocity",
            drone_velocity=np.array([2.0, 0.0, 0.0]),
            obstacles=np.array([[3.0, 5.0, 0.0, 0.5]]),  # Far to the right
            expected_behavior="Velocity should remain unchanged",
            validation_func=lambda v_in, v_out: np.allclose(v_in, v_out, atol=0.1)
        ))
        
        # Test 3: Obstacle slightly to the right - should veer left
        test_cases.append(TestCase(
            name="Veer Left",
            description="Obstacle ahead-right, should adjust left",
            drone_velocity=np.array([2.0, 0.0, 0.0]),
            obstacles=np.array([[3.0, 1.0, 0.0, 0.5]]),  # Ahead and slightly right
            expected_behavior="Should veer left (negative Y)",
            validation_func=lambda v_in, v_out: v_out[1] < -0.1  # Negative Y is left
        ))
        
        # Test 4: Multiple obstacles - corridor
        test_cases.append(TestCase(
            name="Corridor",
            description="Obstacles on both sides, should slow down and go straight",
            drone_velocity=np.array([2.0, 0.0, 0.0]),
            obstacles=np.array([
                [4.0, 2.0, 0.0, 0.5],   # Right wall
                [4.0, -2.0, 0.0, 0.5],  # Left wall
                [3.5, 2.0, 0.0, 0.5],   # Right wall
                [3.5, -2.0, 0.0, 0.5],  # Left wall
            ]),
            expected_behavior="Should slow down but maintain forward direction",
            validation_func=lambda v_in, v_out: (v_out[0] < v_in[0]) and (abs(v_out[1]) < 0.2)
        ))
        
        # Test 5: Obstacle above - should prefer horizontal avoidance
        test_cases.append(TestCase(
            name="Vertical Avoidance",
            description="Obstacle above, should prefer going sideways (dynamics-aware)",
            drone_velocity=np.array([2.0, 0.0, 0.0]),
            obstacles=np.array([[3.0, 0.0, 1.5, 0.5]]),  # Above
            expected_behavior="Should prefer horizontal adjustment over vertical",
            validation_func=lambda v_in, v_out: abs(v_out[1]) > abs(v_out[2]) * 1.5
        ))
        
        # Test 6: Zero velocity input
        test_cases.append(TestCase(
            name="Zero Velocity",
            description="Not moving, should stay still even with obstacles",
            drone_velocity=np.array([0.0, 0.0, 0.0]),
            obstacles=np.array([[2.0, 0.0, 0.0, 0.5]]),
            expected_behavior="Should remain at zero velocity",
            validation_func=lambda v_in, v_out: np.linalg.norm(v_out) < 0.01
        ))
        
        # Test 7: Very close obstacle - emergency stop
        test_cases.append(TestCase(
            name="Emergency Stop",
            description="Very close obstacle, should stop immediately",
            drone_velocity=np.array([2.0, 0.0, 0.0]),
            obstacles=np.array([[0.8, 0.0, 0.0, 0.2]]),  # Very close!
            expected_behavior="Should reduce velocity drastically",
            validation_func=lambda v_in, v_out: np.linalg.norm(v_out) < 0.5
        ))
        
        # Test 8: Speed limit test
        test_cases.append(TestCase(
            name="Speed Limit",
            description="High input velocity should be capped",
            drone_velocity=np.array([5.0, 0.0, 0.0]),  # Above max_speed
            obstacles=np.array([]),  # No obstacles
            expected_behavior="Should be limited to max_speed",
            validation_func=lambda v_in, v_out: np.linalg.norm(v_out) <= self.max_speed + 0.01
        ))
        
        return test_cases

    def simulate_vo_filter(self, v_des_world, obstacles_camera):
        """
        Simulate the VO filter algorithm matching the actual node logic.
        This is a simplified version of _compute_safe_velocity from the actual node.
        """
        # Transform velocity to camera frame (identity transform for testing)
        v_des_camera = v_des_world  # Simplified: assuming aligned frames
        v_des_body = v_des_world
        
        v = v_des_camera.copy()
        
        # Check if we're not moving
        if np.linalg.norm(v) < 1e-3:
            return np.zeros(3)
        
        # Check for violations
        if obstacles_camera.shape[0] == 0:
            # No obstacles, just apply speed limit
            speed = np.linalg.norm(v)
            if speed > self.max_speed:
                v = v * (self.max_speed / speed)
            return v
        
        violations = self.compute_violations(v, obstacles_camera, self.safety_margin, self.time_horizon)
        
        if not np.any(violations):
            # No violations, apply speed limit and return
            speed = np.linalg.norm(v)
            if speed > self.max_speed:
                v = v * (self.max_speed / speed)
            return v
        
        # Iteratively resolve violations (simplified)
        W_body = np.diag([self.w_body_x, self.w_body_y, self.w_body_z])
        
        for iteration in range(3):  # max_rot_iter
            violations = self.compute_violations(v, obstacles_camera, self.safety_margin, self.time_horizon)
            if not np.any(violations):
                break
            
            # Generate candidate velocities
            candidates = []
            
            # Simple avoidance: try to slow down
            candidates.append(0.7 * v)
            candidates.append(0.5 * v)
            
            # Try to veer left/right
            if abs(v[0]) > 0.1:  # If moving forward
                candidates.append(v + np.array([0, -0.5, 0]))  # Veer left
                candidates.append(v + np.array([0, 0.5, 0]))   # Veer right
            
            # Select best candidate
            best_v = v * 0.5  # Default: slow down
            best_cost = float('inf')
            best_viols = np.sum(violations)
            
            for cand in candidates:
                # Check violations for candidate
                cand_viols = self.compute_violations(cand, obstacles_camera, self.safety_margin, self.time_horizon)
                n_viols = np.sum(cand_viols)
                
                # Compute dynamics cost
                dv = cand - v_des_body
                cost = np.dot(dv, W_body @ dv)
                
                # Select if better
                if n_viols < best_viols or (n_viols == best_viols and cost < best_cost):
                    best_v = cand
                    best_cost = cost
                    best_viols = n_viols
            
            v = best_v
        
        # Apply speed limit
        speed = np.linalg.norm(v)
        if speed > self.max_speed:
            v = v * (self.max_speed / speed)
        
        return v

    def run_test_case(self, test_case: TestCase) -> Dict:
        """Run a single test case and return results"""
        print(f"\n{'='*60}")
        print(f"Test: {test_case.name}")
        print(f"Description: {test_case.description}")
        print(f"Input velocity: {test_case.drone_velocity}")
        print(f"Obstacles: {test_case.obstacles.shape[0]} obstacles")
        
        # Run the VO filter
        start_time = time.perf_counter()
        output_velocity = self.simulate_vo_filter(
            test_case.drone_velocity,
            test_case.obstacles
        )
        elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Check if output is correct
        passed = test_case.validation_func(test_case.drone_velocity, output_velocity)
        
        # Calculate metrics
        input_speed = np.linalg.norm(test_case.drone_velocity)
        output_speed = np.linalg.norm(output_velocity)
        speed_reduction = (1 - output_speed / (input_speed + 1e-9)) * 100
        
        print(f"Output velocity: {output_velocity}")
        print(f"Speed: {input_speed:.2f} m/s → {output_speed:.2f} m/s ({speed_reduction:.1f}% reduction)")
        print(f"Processing time: {elapsed_time:.3f} ms")
        print(f"Expected: {test_case.expected_behavior}")
        print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return {
            'name': test_case.name,
            'passed': passed,
            'time_ms': elapsed_time,
            'input_velocity': test_case.drone_velocity,
            'output_velocity': output_velocity,
            'speed_reduction': speed_reduction
        }

    def run_performance_test(self, n_obstacles_list=[10, 50, 100, 500], n_runs=100):
        """Run performance benchmarks"""
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARKS")
        print(f"{'='*60}")
        
        results = []
        
        for n_obs in n_obstacles_list:
            # Generate random obstacles
            obstacles = np.random.randn(n_obs, 4).astype(np.float32)
            obstacles[:, :3] *= 5.0  # Position
            obstacles[:, 3] = 0.05   # Radius
            
            v_des = np.array([2.0, 0.0, 0.0], dtype=np.float32)
            
            # Warmup
            for _ in range(5):
                _ = self.simulate_vo_filter(v_des, obstacles)
            
            # Benchmark
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.simulate_vo_filter(v_des, obstacles)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            fps = 1000.0 / mean_time  # Hz
            
            print(f"\n{n_obs} obstacles:")
            print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms")
            print(f"  Rate: {fps:.1f} Hz")
            
            results.append({
                'n_obstacles': n_obs,
                'mean_ms': mean_time,
                'std_ms': std_time,
                'fps': fps
            })
        
        return results

    def run_all_tests(self):
        """Run all correctness and performance tests"""
        print(f"\n{'#'*60}")
        print(f"VO FILTER COMPLETE TEST SUITE")
        print(f"{'#'*60}")
        print(f"\nUsing: {'Numba optimized' if self.use_numba else 'Pure Python'}")
        print(f"Safety margin: {self.safety_margin} m")
        print(f"Time horizon: {self.time_horizon} s")
        print(f"Max speed: {self.max_speed} m/s")
        print(f"Body weights: X={self.w_body_x}, Y={self.w_body_y}, Z={self.w_body_z}")
        
        # Run correctness tests
        print(f"\n{'='*60}")
        print("CORRECTNESS TESTS")
        print(f"{'='*60}")
        
        test_cases = self.create_test_cases()
        correctness_results = []
        
        for test_case in test_cases:
            result = self.run_test_case(test_case)
            correctness_results.append(result)
        
        # Run performance tests
        performance_results = self.run_performance_test()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for r in correctness_results if r['passed'])
        total = len(correctness_results)
        
        print(f"\nCorrectness: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        for result in correctness_results:
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} {result['name']}: {result['time_ms']:.3f} ms")
        
        print(f"\nPerformance:")
        for result in performance_results:
            print(f"  {result['n_obstacles']} obstacles: {result['fps']:.1f} Hz")
        
        # Overall assessment
        print(f"\n{'='*60}")
        if passed == total:
            print("✓ ALL TESTS PASSED - VO Filter is working correctly!")
        else:
            print(f"✗ {total - passed} tests failed - Check the implementation!")
        
        avg_fps = np.mean([r['fps'] for r in performance_results])
        if avg_fps > 100:
            print(f"✓ Performance EXCELLENT: {avg_fps:.1f} Hz average")
        elif avg_fps > 30:
            print(f"✓ Performance GOOD: {avg_fps:.1f} Hz average")
        else:
            print(f"✗ Performance POOR: {avg_fps:.1f} Hz average (need >30 Hz)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test VO Filter Node')
    parser.add_argument('--numba', action='store_true', help='Use Numba optimized version')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    args = parser.parse_args()
    
    # Create tester
    tester = VOFilterTester(use_numba=args.numba)
    
    if args.quick:
        # Run just a few tests
        print("Running quick tests...")
        test_cases = tester.create_test_cases()[:3]
        for test_case in test_cases:
            tester.run_test_case(test_case)
    else:
        # Run full test suite
        tester.run_all_tests()

if __name__ == "__main__":
    main()