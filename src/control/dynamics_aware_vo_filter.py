#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamics-aware 3D Velocity-Obstacle (VO) safety filter for UAVs.

- Subscribes: desired velocity (TwistStamped), point cloud, odom
- Publishes: corrected/safe velocity (TwistStamped)
- Core idea: if the desired velocity is inside any 3D collision cone within a time horizon,
  generate candidate "escape" velocities on the cone boundaries and pick the one
  that minimizes a body-aligned weighted norm ||v - v_des||_W (dynamics-aware closeness).
- If v_des == 0 and you're safe, output stays exactly 0 (no APF-style push-away).

Notes:
- Assumes the point cloud is in the same frame as odom (or already transformed).
  If not, add a TF transform to 'odom'.
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
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import pcl # Added for PCL filtering
from numba import jit # Added for Numba JIT compilation

class DynamicsAwareVOFilter3D(object):
    def __init__(self):
        rospy.init_node('dynamics_aware_vo_filter_3d')

        # Topics
        self.input_twist_topic = rospy.get_param('~input_twist_topic', '/uav/attractive_velocity')
        self.output_twist_topic = rospy.get_param('~output_twist_topic', '/uav/safe_velocity')
        self.cloud_topic = rospy.get_param('~cloud_topic', '/camera/depth/points')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.world_frame = rospy.get_param('~world_frame', 'odom')

        # VO / safety params - cast to float32 for Numba
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
        self.w_body_z = np.float32(rospy.get_param('~w_body_z', 3.0))

        # Speed limit (total velocity magnitude)
        self.max_speed = np.float32(rospy.get_param('~max_speed', 2.5))

        # State - Using float32 for Numba compatibility
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.R_wb = np.eye(3, dtype=np.float32)
        self.cloud_pts = None
        self.cloud_lock = threading.Lock()
        
        self.R_cb = np.eye(3, dtype=np.float32)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # IO
        self.pub = rospy.Publisher(self.output_twist_topic, TwistStamped, queue_size=10)
        rospy.Subscriber(self.input_twist_topic, TwistStamped, self.twist_cb, queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=20)

        rospy.loginfo("DynamicsAwareVOFilter3D ready. Subscribed to %s, %s, %s",
                      self.input_twist_topic, self.cloud_topic, self.odom_topic)

    # ---------- Callbacks ----------

    def odom_cb(self, msg):
        self.pos[:] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        self.vel[:] = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.float32)
        q = msg.pose.pose.orientation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        self.R_wb = R.astype(np.float32)

    def cloud_cb(self, msg):
        try:
            points_list = point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
            if not points_list:
                with self.cloud_lock:
                    self.cloud_pts = None
                return

            p = pcl.PointCloud(np.array(points_list, dtype=np.float32))

            vox = p.make_voxel_grid_filter()
            vox.set_leaf_size(self.voxel_size, self.voxel_size, self.voxel_size)
            cloud_filtered = vox.filter()
            pts = cloud_filtered.to_array()

            with self.cloud_lock:
                if pts.shape[0] > 0:
                    d2 = np.sum(pts * pts, axis=1)
                    mask = (d2 >= self.min_range**2) & (d2 <= self.max_range**2)
                    self.cloud_pts = pts[mask]
                else:
                    self.cloud_pts = None
        except Exception as e:
            rospy.logwarn_throttle(2.0, "PCL filtering failed: %s", str(e))

    def twist_cb(self, msg):
        v_des_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32)
        v_des_body = self.R_wb.T @ v_des_world
        v_des_camera = self.R_cb.T @ v_des_body
        
        with self.cloud_lock:
            pts = None if self.cloud_pts is None else self.cloud_pts.copy()
        
        obstacles = self._spheres_from_cloud_camera(pts, k=self.max_obstacles)
        
        v_safe_camera = self._compute_safe_velocity_camera(v_des_camera, v_des_body, obstacles)
        
        v_safe_body = self.R_cb @ v_safe_camera
        v_safe_world = self.R_wb @ v_safe_body
        v_safe_world = self._clip_speed(v_safe_world)

        out = TwistStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.twist.linear.x, out.twist.linear.y, out.twist.linear.z = v_safe_world.tolist()
        out.twist.angular.z = msg.twist.angular.z
        self.pub.publish(out)
        
        if np.linalg.norm(v_des_world - v_safe_world) > 0.01:
            rospy.loginfo_throttle(1.0, f"VO corrected velocity. Obstacles: {obstacles.shape[0]}")

    # ---------- Core: dynamics-aware 3D VO ----------

    def _compute_safe_velocity_camera(self, v_des_camera, v_des_body, obstacles):
        v = v_des_camera.copy()
        speed = np.linalg.norm(v)
        if speed < 1e-3:
            return np.zeros(3, dtype=np.float32)

        W_body = np.diag([self.w_body_x, self.w_body_y, self.w_body_z]).astype(np.float32)

        viol_indices = self._violations_camera_jit(v, obstacles, self.safety_margin, self.time_horizon)
        if not viol_indices:
            return v

        base_des_camera = v.copy()
        base_des_body = v_des_body.copy()
        
        for _ in range(self.max_rot_iter):
            viol_indices = self._violations_camera_jit(v, obstacles, self.safety_margin, self.time_horizon)
            if not viol_indices:
                break

            viols = obstacles[viol_indices]
            candidates = []
            for i in range(viols.shape[0]):
                obs = viols[i]
                cand = self._rotate_to_cone_boundary_camera(v, obs)
                if cand is not None:
                    candidates.append(cand)

            candidates.append(0.7 * v)
            # As per request, the zero-velocity candidate is removed to prevent the drone from stopping completely.
            # candidates.append(np.zeros(3))

            best = None
            best_cost = float('inf')
            best_viol_count = 1e9

            for c_camera in candidates:
                c_viol_indices = self._violations_camera_jit(c_camera, obstacles, self.safety_margin, self.time_horizon)
                num_violations = len(c_viol_indices)
                c_body = self.R_cb @ c_camera
                cost = self._dyn_cost_body(c_body, base_des_body, W_body)

                if (num_violations < best_viol_count) or (num_violations == best_viol_count and cost < best_cost):
                    best = c_camera
                    best_cost = cost
                    best_viol_count = num_violations

            if best is None:
                break
            v = best
        return v

    @staticmethod
    @jit(nopython=True)
    def _violations_camera_jit(v, obstacles, safety_margin, time_horizon):
        viol_indices = []
        if obstacles.shape[0] == 0:
            return viol_indices
        
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return viol_indices

        for i in range(obstacles.shape[0]):
            p = obstacles[i, :3]
            R = obstacles[i, 3] + safety_margin
            d = np.linalg.norm(p)

            if d <= R:
                viol_indices.append(i)
                continue

            p_hat = p / d
            closing = np.dot(v, p_hat)
            if closing <= 0.0:
                continue

            ttc = (d - R) / closing
            if ttc <= time_horizon:
                ratio = R / d
                clipped_ratio = max(np.float32(0.0), min(ratio, np.float32(1.0)))
                theta = np.arcsin(clipped_ratio)
                
                dot_product = np.dot(v / v_norm, p_hat)
                clipped_dot = max(np.float32(-1.0), min(dot_product, np.float32(1.0)))
                ang = np.arccos(clipped_dot)

                if ang <= theta:
                    viol_indices.append(i)
        return viol_indices

    def _rotate_to_cone_boundary_camera(self, v, obs):
        p = obs[:3]
        R = obs[3] + self.safety_margin
        d = np.linalg.norm(p)
        if d <= R or np.linalg.norm(v) < 1e-6:
            return None

        p_hat = p / d
        speed = np.linalg.norm(v)
        v_hat = v / speed
        
        ratio = R / d
        clipped_ratio = max(np.float32(0.0), min(ratio, np.float32(1.0)))
        theta = np.arcsin(clipped_ratio)

        dot_product = np.dot(v_hat, p_hat)
        clipped_dot = max(np.float32(-1.0), min(dot_product, np.float32(1.0)))
        ang = np.arccos(clipped_dot)

        if ang > theta + self.boundary_eps:
            return v

        axis = np.cross(v, p)
        na = np.linalg.norm(axis)
        if na < 1e-9:
            t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(t, p_hat)) > 0.9:
                t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = np.cross(p_hat, t)
            na = np.linalg.norm(axis)
            if na < 1e-9: return None
        axis /= na

        delta = (theta - ang) + self.boundary_eps
        v_rot = self._rodrigues(v, axis, delta)
        v_rot = (v_rot / (np.linalg.norm(v_rot) + 1e-9)) * speed
        return v_rot

    @staticmethod
    @jit(nopython=True)
    def _rodrigues(vec, k_hat, angle):
        v = vec
        k = k_hat
        c = np.cos(angle)
        s = np.sin(angle)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (np.float32(1.0) - c)

    @staticmethod
    @jit(nopython=True)
    def _dyn_cost_body(v_body, v_des_body, W_body):
        dv = (v_body - v_des_body).reshape(3, 1)
        # The result of the dot product is a 1x1 matrix, so we extract the scalar value
        return np.dot(np.dot(dv.T, W_body), dv)[0, 0]

    def _clip_speed(self, v):
        v = np.array(v, dtype=np.float32)
        speed = np.linalg.norm(v)
        if speed > self.max_speed:
            v = v * (self.max_speed / speed)
        return v

    def _spheres_from_cloud_camera(self, pts, k=16):
        if pts is None or pts.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32)
        
        d2 = np.sum(pts * pts, axis=1)
        mask = (d2 >= self.min_range**2) & (d2 <= self.max_range**2)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.empty((0, 4), dtype=np.float32)

        take = min(k, idx.size)
        nearest_idx_in_idx = np.argpartition(d2[idx], take - 1)[:take]
        nearest_idx = idx[nearest_idx_in_idx]

        obs_data = np.zeros((len(nearest_idx), 4), dtype=np.float32)
        for i, pt_idx in enumerate(nearest_idx):
            obs_data[i, :3] = pts[pt_idx]
            obs_data[i, 3] = self.point_obstacle_radius
        return obs_data

if __name__ == '__main__':
    try:
        node = DynamicsAwareVOFilter3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass