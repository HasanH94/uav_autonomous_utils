#!/usr/bin/env python3
"""
Standalone test for VO filter logic without ROS
"""

import numpy as np
import math
import sys
import os

# Mock ROS modules
class MockRospy:
    @staticmethod
    def init_node(name):
        print(f"Mock: Initialized node '{name}'")
    
    @staticmethod
    def get_param(param, default):
        return default
    
    @staticmethod
    def loginfo(msg):
        print(f"INFO: {msg}")
    
    @staticmethod
    def logwarn(msg):
        print(f"WARN: {msg}")
    
    @staticmethod
    def logwarn_throttle(rate, msg, *args):
        print(f"WARN: {msg % args if args else msg}")
    
    @staticmethod
    def logdebug(msg):
        print(f"DEBUG: {msg}")
    
    class Time:
        @staticmethod
        def now():
            return None
    
    class Duration:
        def __init__(self, secs):
            self.secs = secs

# Mock tf.transformations
def quaternion_matrix(q):
    """Simple quaternion to rotation matrix conversion"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 0],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 0],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), 0],
        [0, 0, 0, 1]
    ])

# Replace imports
sys.modules['rospy'] = MockRospy
sys.modules['tf.transformations'] = type('module', (), {'quaternion_matrix': quaternion_matrix})()

# Now we can import our modified VO filter logic
class DynamicsAwareVOFilter3D:
    """Simplified version for testing"""
    
    def __init__(self):
        # Parameters
        self.time_horizon = 3.0
        self.safety_margin = 1.0
        self.point_obstacle_radius = 0.05
        self.max_obstacles = 16
        self.min_range = 0.30
        self.max_range = 12.0
        self.max_rot_iter = 3
        self.boundary_eps = 1e-3
        
        # Dynamics weights (body frame)
        self.w_body_x = 2.0  # forward/backward (harder)
        self.w_body_y = 1.0  # left/right (easiest)
        self.w_body_z = 4.0  # up/down (hardest)
        
        self.max_speed = 3.0
        
        # State
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.R_wb = np.eye(3)  # body->world
        self.R_cb = np.eye(3)  # camera->body (identity for aligned camera)
        
        print("VO Filter initialized (test mode)")
    
    def set_state(self, pos, quaternion):
        """Set drone position and orientation"""
        self.pos = np.array(pos)
        R = quaternion_matrix(quaternion)[:3, :3]
        self.R_wb = R
        
    def test_scenario(self, v_des_world, obstacles_camera):
        """Test the VO filter with given velocity and obstacles"""
        print(f"\n--- Test Scenario ---")
        print(f"Desired velocity (world): {v_des_world}")
        print(f"Obstacles in camera frame: {len(obstacles_camera)}")
        
        # Transform velocity to body then camera frame
        v_des_body = self.R_wb.T @ v_des_world
        v_des_camera = self.R_cb.T @ v_des_body
        
        print(f"Desired velocity (body): {v_des_body}")
        print(f"Desired velocity (camera): {v_des_camera}")
        
        # Create obstacle list
        obstacles = []
        for pt in obstacles_camera:
            obstacles.append({'p': np.array(pt), 'r': self.point_obstacle_radius})
        
        # Compute safe velocity
        v_safe_camera = self._compute_safe_velocity_camera(v_des_camera, v_des_body, obstacles)
        
        # Transform back
        v_safe_body = self.R_cb @ v_safe_camera
        v_safe_world = self.R_wb @ v_safe_body
        v_safe_world = self._clip_speed(v_safe_world)
        
        print(f"Safe velocity (camera): {v_safe_camera}")
        print(f"Safe velocity (body): {v_safe_body}")
        print(f"Safe velocity (world): {v_safe_world}")
        
        return v_safe_world
    
    def _compute_safe_velocity_camera(self, v_des_camera, v_des_body, obstacles):
        """Simplified safe velocity computation"""
        v = v_des_camera.copy()
        
        speed = np.linalg.norm(v)
        if speed < 1e-3:
            return np.zeros(3)
        
        W_body = np.diag([self.w_body_x, self.w_body_y, self.w_body_z])
        
        # Check for violations
        if not self._violations_camera(v, obstacles):
            print("No violations detected")
            return v
        
        print("Violations detected, computing safe velocity...")
        
        # Simple resolution: slow down
        for scale in [0.7, 0.5, 0.3, 0.0]:
            v_test = scale * v
            if not self._violations_camera(v_test, obstacles):
                print(f"Safe at {scale*100:.0f}% speed")
                return v_test
        
        return np.zeros(3)
    
    def _violations_camera(self, v, obstacles):
        """Check for collisions in camera frame"""
        viols = []
        v_norm = np.linalg.norm(v) + 1e-9
        
        for obs in obstacles:
            p = obs['p']  # Already in camera frame
            d = np.linalg.norm(p)
            R = obs['r'] + self.safety_margin
            
            if d <= R:
                print(f"  Inside safety bubble! d={d:.2f}, R={R:.2f}")
                viols.append(obs)
                continue
            
            p_hat = p / d
            closing = float(np.dot(v, p_hat))
            
            if closing <= 0.0:
                continue
            
            ttc = (d - R) / max(closing, 1e-6)
            if ttc <= self.time_horizon:
                theta = math.asin(min(1.0, R / d))
                ang = math.acos(np.clip(np.dot(v / v_norm, p_hat), -1.0, 1.0))
                if ang <= theta:
                    print(f"  Collision predicted: d={d:.2f}m, ttc={ttc:.2f}s")
                    viols.append(obs)
        
        return viols
    
    def _clip_speed(self, v):
        """Clip total velocity magnitude"""
        v = np.array(v, dtype=float)
        speed = np.linalg.norm(v)
        if speed > self.max_speed:
            v = v * (self.max_speed / speed)
        return v


def run_tests():
    """Run test scenarios"""
    vo_filter = DynamicsAwareVOFilter3D()
    
    # Test 1: No obstacles
    print("\n=== Test 1: No obstacles ===")
    vo_filter.set_state([0, 0, 2], [0, 0, 0, 1])  # At origin, facing forward
    v_des = np.array([1.0, 0.0, 0.0])  # Forward at 1 m/s
    obstacles = []
    v_safe = vo_filter.test_scenario(v_des, obstacles)
    assert np.allclose(v_safe, v_des), "Should return original velocity"
    
    # Test 2: Obstacle directly ahead in camera frame
    print("\n=== Test 2: Obstacle ahead (camera frame) ===")
    vo_filter.set_state([0, 0, 2], [0, 0, 0, 1])
    v_des = np.array([2.0, 0.0, 0.0])  # Forward at 2 m/s
    obstacles = [[3.0, 0.0, 0.0]]  # 3m ahead in camera frame
    v_safe = vo_filter.test_scenario(v_des, obstacles)
    print(f"Speed reduced from {np.linalg.norm(v_des):.2f} to {np.linalg.norm(v_safe):.2f}")
    
    # Test 3: Obstacle to the side (should not affect forward motion)
    print("\n=== Test 3: Obstacle to the side ===")
    vo_filter.set_state([0, 0, 2], [0, 0, 0, 1])
    v_des = np.array([1.0, 0.0, 0.0])  # Forward
    obstacles = [[0.0, 3.0, 0.0]]  # 3m to the left in camera frame
    v_safe = vo_filter.test_scenario(v_des, obstacles)
    assert np.allclose(v_safe, v_des), "Side obstacle shouldn't affect forward motion"
    
    # Test 4: Very close obstacle
    print("\n=== Test 4: Very close obstacle ===")
    vo_filter.set_state([0, 0, 2], [0, 0, 0, 1])
    v_des = np.array([1.0, 0.0, 0.0])
    obstacles = [[0.5, 0.0, 0.0]]  # Only 0.5m ahead!
    v_safe = vo_filter.test_scenario(v_des, obstacles)
    print(f"Very close obstacle: speed = {np.linalg.norm(v_safe):.2f}")
    
    # Test 5: Drone rotated 90 degrees (facing left)
    print("\n=== Test 5: Drone rotated 90° (facing left) ===")
    # Quaternion for 90° yaw rotation: [0, 0, 0.707, 0.707]
    vo_filter.set_state([0, 0, 2], [0, 0, 0.707, 0.707])
    v_des = np.array([0.0, 1.0, 0.0])  # Moving left in world = forward in body
    obstacles = [[3.0, 0.0, 0.0]]  # Obstacle ahead in camera (which is now facing left)
    v_safe = vo_filter.test_scenario(v_des, obstacles)
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    run_tests()