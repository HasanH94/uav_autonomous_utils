#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Dynamics-aware 3D Velocity-Obstacle (VO) safety filter for UAVs.

Key optimizations:
- Uses Open3D for fast point cloud processing (better Python support than PCL)
- Proper Numba JIT compilation with pre-allocated arrays
- Minimal data copying between formats
- Vectorized operations where possible
"""

import rospy
import numpy as np
import math
import threading
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from tf.transformations import quaternion_matrix
import tf2_ros
from numba import jit, njit, prange
import numba

# Try to import Open3D, fall back to numpy if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    rospy.logwarn("Open3D not found, using numpy fallback (slower). Install with: pip3 install open3d")

# Pre-compile Numba functions with specific signatures
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

@njit(fastmath=True, cache=True)
def rotate_to_cone_boundary(v_camera, obs_p, obs_r, safety_margin, boundary_eps):
    """Rotate velocity to cone boundary - fully Numba compiled."""
    p = obs_p
    R = obs_r + safety_margin
    d = np.linalg.norm(p)
    
    if d <= R or np.linalg.norm(v_camera) < 1e-6:
        return v_camera  # Can't rotate out
    
    p_hat = p / d
    speed = np.linalg.norm(v_camera)
    v_hat = v_camera / speed
    
    # Target angle (cone boundary)
    theta = np.arcsin(min(1.0, R / d))
    
    # Current angle
    cos_angle = np.dot(v_hat, p_hat)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    if angle > theta + boundary_eps:
        return v_camera  # Already outside
    
    # Rotation axis
    axis = np.array([
        v_camera[1]*p[2] - v_camera[2]*p[1],
        v_camera[2]*p[0] - v_camera[0]*p[2],
        v_camera[0]*p[1] - v_camera[1]*p[0]
    ])
    
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # Vectors are parallel, pick perpendicular axis
        if abs(p_hat[0]) < 0.9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - p_hat * np.dot(axis, p_hat)
    
    # Rotate just outside cone
    delta = (theta - angle) + boundary_eps
    v_rot = rodrigues_rotation(v_camera, axis, -delta)
    
    # Maintain speed
    return v_rot * (speed / (np.linalg.norm(v_rot) + 1e-9))

class DynamicsAwareVOFilter3D(object):
    def __init__(self):
        rospy.init_node('dynamics_aware_vo_filter_3d')

        # Topics
        self.input_twist_topic = rospy.get_param('~input_twist_topic', '/uav/attractive_velocity')
        self.output_twist_topic = rospy.get_param('~output_twist_topic', '/uav/safe_velocity')
        self.cloud_topic = rospy.get_param('~cloud_topic', '/camera/depth/points')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.world_frame = rospy.get_param('~world_frame', 'odom')

        # VO / safety params
        self.time_horizon = np.float32(rospy.get_param('~time_horizon', 3.0))
        self.safety_margin = np.float32(rospy.get_param('~safety_margin', 1.0))
        self.point_obstacle_radius = np.float32(rospy.get_param('~point_obstacle_radius', 0.05))
        self.max_obstacles = rospy.get_param('~max_obstacles', 16)
        self.min_range = np.float32(rospy.get_param('~min_range', 0.30))
        self.max_range = np.float32(rospy.get_param('~max_range', 12.0))
        self.max_rot_iter = rospy.get_param('~max_rot_iter', 3)
        self.boundary_eps = np.float32(rospy.get_param('~boundary_eps', 1e-3))
        self.voxel_size = np.float32(rospy.get_param('~voxel_size', 0.20))

        # Dynamics-aware closeness (body-aligned weights)
        self.w_body_x = np.float32(rospy.get_param('~w_body_x', 2.0))
        self.w_body_y = np.float32(rospy.get_param('~w_body_y', 1.0))
        self.w_body_z = np.float32(rospy.get_param('~w_body_z', 4.0))

        # Speed limit
        self.max_speed = np.float32(rospy.get_param('~max_speed', 3.0))

        # State
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.R_wb = np.eye(3, dtype=np.float32)
        self.cloud_pts = None
        self.cloud_lock = threading.Lock()
        
        # Camera to body transformation (identity if camera aligned with body)
        self.R_cb = np.eye(3, dtype=np.float32)
        
        # Pre-allocate arrays for performance
        self.obstacle_buffer = np.zeros((self.max_obstacles, 4), dtype=np.float32)

        # IO
        self.pub = rospy.Publisher(self.output_twist_topic, TwistStamped, queue_size=10)
        rospy.Subscriber(self.input_twist_topic, TwistStamped, self.twist_cb, queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=20)

        rospy.loginfo("DynamicsAwareVOFilter3D (Optimized) ready. Using %s for point cloud processing",
                      "Open3D" if HAS_OPEN3D else "NumPy")

    def odom_cb(self, msg):
        self.pos[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.vel[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        q = msg.pose.pose.orientation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        self.R_wb = R.astype(np.float32)

    def cloud_cb(self, msg):
        """Process point cloud with Open3D for speed or NumPy fallback."""
        try:
            # Extract points efficiently
            points = []
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = p
                # Early range filtering
                d_sq = x*x + y*y + z*z
                if self.min_range**2 <= d_sq <= self.max_range**2:
                    points.append([x, y, z])
                    if len(points) >= 10000:  # Hard limit for safety
                        break
            
            if not points:
                with self.cloud_lock:
                    self.cloud_pts = None
                return
            
            pts = np.array(points, dtype=np.float32)
            
            # Voxel downsampling
            if HAS_OPEN3D and pts.shape[0] > 100:
                # Use Open3D for fast voxel downsampling
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd_down = pcd.voxel_down_sample(self.voxel_size)
                pts_filtered = np.asarray(pcd_down.points, dtype=np.float32)
            else:
                # NumPy fallback - simple voxel grid
                pts_filtered = self._voxel_downsample_numpy(pts, self.voxel_size)
            
            with self.cloud_lock:
                self.cloud_pts = pts_filtered
                
        except Exception as e:
            rospy.logwarn_throttle(2.0, "PointCloud processing failed: %s", str(e))

    def _voxel_downsample_numpy(self, points, voxel_size):
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

    def twist_cb(self, msg):
        v_des_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32)
        
        # Transform to body then camera frame
        v_des_body = self.R_wb.T @ v_des_world
        v_des_camera = self.R_cb.T @ v_des_body
        
        # Get obstacles
        with self.cloud_lock:
            pts = self.cloud_pts.copy() if self.cloud_pts is not None else None
        
        obstacles = self._get_nearest_obstacles(pts)
        
        # Compute safe velocity
        v_safe_camera = self._compute_safe_velocity(v_des_camera, v_des_body, obstacles)
        
        # Transform back to world
        v_safe_body = self.R_cb @ v_safe_camera
        v_safe_world = self.R_wb @ v_safe_body
        
        # Apply speed limit
        speed = np.linalg.norm(v_safe_world)
        if speed > self.max_speed:
            v_safe_world *= self.max_speed / speed

        # Publish
        out = TwistStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.twist.linear.x, out.twist.linear.y, out.twist.linear.z = v_safe_world.tolist()
        out.twist.angular.z = msg.twist.angular.z
        self.pub.publish(out)
        
        if np.linalg.norm(v_des_world - v_safe_world) > 0.01:
            rospy.loginfo_throttle(1.0, f"VO corrected: {len(obstacles)} obstacles")

    def _get_nearest_obstacles(self, pts):
        """Get nearest obstacles as Nx4 array [x,y,z,radius]."""
        if pts is None or pts.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        # Compute distances
        distances = np.linalg.norm(pts, axis=1)
        
        # Take nearest k points
        k = min(self.max_obstacles, pts.shape[0])
        nearest_idx = np.argpartition(distances, k-1)[:k] if k < pts.shape[0] else np.arange(pts.shape[0])
        
        # Fill obstacle array
        obstacles = np.zeros((len(nearest_idx), 4), dtype=np.float32)
        obstacles[:, :3] = pts[nearest_idx]
        obstacles[:, 3] = self.point_obstacle_radius
        
        return obstacles

    def _compute_safe_velocity(self, v_des_camera, v_des_body, obstacles):
        """Main VO algorithm using optimized Numba functions."""
        v = v_des_camera.copy()
        
        if np.linalg.norm(v) < 1e-3:
            return np.zeros(3, dtype=np.float32)
        
        # Check for violations using vectorized Numba function
        violations = compute_violations_vectorized(v, obstacles, self.safety_margin, self.time_horizon)
        
        if not np.any(violations):
            return v
        
        # Body frame weight matrix
        W_body = np.diag([self.w_body_x, self.w_body_y, self.w_body_z]).astype(np.float32)
        
        # Iteratively resolve violations
        for iteration in range(self.max_rot_iter):
            violations = compute_violations_vectorized(v, obstacles, self.safety_margin, self.time_horizon)
            if not np.any(violations):
                break
            
            # Generate candidate velocities
            candidates = []
            
            # For each violating obstacle, rotate to boundary
            viol_indices = np.where(violations)[0]
            for idx in viol_indices[:5]:  # Limit candidates for speed
                obs = obstacles[idx]
                v_rot = rotate_to_cone_boundary(
                    v, obs[:3], obs[3], self.safety_margin, self.boundary_eps
                )
                candidates.append(v_rot)
            
            # Add conservative options
            candidates.append(0.7 * v)
            candidates.append(0.5 * v)
            
            # Select best candidate
            best_v = v
            best_cost = float('inf')
            best_viols = len(viol_indices)
            
            for cand in candidates:
                cand_viols = compute_violations_vectorized(cand, obstacles, self.safety_margin, self.time_horizon)
                n_viols = np.sum(cand_viols)
                
                # Compute dynamics cost in body frame
                cand_body = self.R_cb @ cand
                dv = cand_body - v_des_body
                cost = np.dot(dv, W_body @ dv)
                
                # Prefer fewer violations, then lower cost
                if n_viols < best_viols or (n_viols == best_viols and cost < best_cost):
                    best_v = cand
                    best_cost = cost
                    best_viols = n_viols
            
            v = best_v
        
        return v

if __name__ == '__main__':
    try:
        node = DynamicsAwareVOFilter3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass