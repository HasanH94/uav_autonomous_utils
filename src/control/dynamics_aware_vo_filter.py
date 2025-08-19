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
        self.time_horizon = rospy.get_param('~time_horizon', 3.0)         # [s] react only if collision < τ
        self.safety_margin = rospy.get_param('~safety_margin', 1.0)      # [m]
        self.point_obstacle_radius = rospy.get_param('~point_obstacle_radius', 0.05)  # [m] inflate a point
        self.max_obstacles = rospy.get_param('~max_obstacles', 16)
        self.min_range = rospy.get_param('~min_range', 0.30)              # [m] ignore too-close returns (self-body)
        self.max_range = rospy.get_param('~max_range', 12.0)              # [m]
        self.max_rot_iter = rospy.get_param('~max_rot_iter', 3)           # VO resolve iterations
        self.boundary_eps = rospy.get_param('~boundary_eps', 1e-3)
        self.voxel_size = rospy.get_param('~voxel_size', 0.20)            # [m] for light downsampling

        # Dynamics-aware closeness (body-aligned weights)
        # Weight preferences in BODY frame: easier to move in Y than X than Z
        self.w_body_x = rospy.get_param('~w_body_x', 2.0)   # forward/backward in body frame
        self.w_body_y = rospy.get_param('~w_body_y', 1.0)   # left/right in body frame (easiest)
        self.w_body_z = rospy.get_param('~w_body_z', 4.0)   # up/down in body frame (hardest)

        # Speed limit (total velocity magnitude)
        self.max_speed = rospy.get_param('~max_speed', 3.0)   # [m/s] max total velocity

        # State
        self.pos = np.zeros(3)  # Position in world frame
        self.vel = np.zeros(3)  # Velocity in world frame (from odom)
        self.R_wb = np.eye(3)   # body->world rotation
        self.cloud_pts = None   # Nx3 numpy array in CAMERA frame
        self.cloud_lock = threading.Lock()
        
        # Camera to body transformation (typical forward-facing camera)
        # Camera: +X forward, +Y left, +Z up (optical convention)
        # Body: +X forward, +Y left, +Z up (FLU convention)
        # If camera is mounted aligned with body, this is identity
        # Adjust based on your actual camera mounting
        self.R_cb = np.eye(3)  # camera-to-body rotation
        
        # TF buffer for transforming point clouds
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
        self.pos[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.vel[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        q = msg.pose.pose.orientation
        # quaternion_matrix returns a 4x4 homogeneous matrix
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        # Standard convention: matrix maps body->world
        self.R_wb = R

    def cloud_cb(self, msg):
        """Store downsampled point cloud in CAMERA frame (no transformation)."""
        pts = []
        # Light voxel filter via dict of voxel indices
        vox = {}
        try:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = p
                # Range gate in camera frame (distance from camera)
                d = x*x + y*y + z*z
                if d < self.min_range**2 or d > self.max_range**2:
                    continue
                # voxel key
                kx = int(math.floor(x / self.voxel_size))
                ky = int(math.floor(y / self.voxel_size))
                kz = int(math.floor(z / self.voxel_size))
                key = (kx, ky, kz)
                if key not in vox:
                    vox[key] = (x, y, z)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "PointCloud2 parsing failed: %s", str(e))

        if vox:
            pts = np.array(list(vox.values()), dtype=np.float32)
            with self.cloud_lock:
                self.cloud_pts = pts

    def twist_cb(self, msg):
        v_des_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=float)
        
        # Transform desired velocity from world to body frame
        v_des_body = self.R_wb.T @ v_des_world
        
        # Transform from body to camera frame for obstacle checking
        v_des_camera = self.R_cb.T @ v_des_body
        
        # Get obstacles in camera frame
        with self.cloud_lock:
            pts = None if self.cloud_pts is None else self.cloud_pts.copy()
        
        obstacles = self._spheres_from_cloud_camera(pts, k=self.max_obstacles)
        
        # Compute safe velocity in camera frame
        v_safe_camera = self._compute_safe_velocity_camera(v_des_camera, v_des_body, obstacles)
        
        # Transform safe velocity back: camera -> body -> world
        v_safe_body = self.R_cb @ v_safe_camera
        v_safe_world = self.R_wb @ v_safe_body
        
        # Apply speed limit to final velocity
        v_safe_world = self._clip_speed(v_safe_world)

        # Publish
        out = TwistStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.twist.linear.x, out.twist.linear.y, out.twist.linear.z = v_safe_world.tolist()
        out.twist.angular.z = msg.twist.angular.z  # pass-through yaw rate
        self.pub.publish(out)
        
        if np.linalg.norm(v_des_world - v_safe_world) > 0.01:
            rospy.loginfo(f"VO Filter: v_des={v_des_world}, v_safe={v_safe_world}, obstacles={len(obstacles)}")
        else:
            rospy.logdebug(f"VO Filter: No correction needed")

    # ---------- Core: dynamics-aware 3D VO ----------

    def _compute_safe_velocity_camera(self, v_des_camera, v_des_body, obstacles):
        """Compute safe velocity in CAMERA frame, using body-frame dynamics weights."""
        v = v_des_camera.copy()

        speed = np.linalg.norm(v)
        if speed < 1e-3:
            # If we're not moving, don't invent motion (keeps goal accuracy)
            return np.zeros(3)

        # Dynamics weights stay in BODY frame
        W_body = np.diag([self.w_body_x, self.w_body_y, self.w_body_z])

        # Quick exit: if no violations, return original velocity
        if not self._violations_camera(v, obstacles):
            return v

        # Iteratively resolve violations
        base_des_camera = v.copy()
        base_des_body = v_des_body.copy()
        
        for _ in range(self.max_rot_iter):
            viols = self._violations_camera(v, obstacles)
            if not viols:
                break

            candidates = []
            # 1) For each violating obstacle, rotate minimally to cone boundary
            for obs in viols:
                cand = self._rotate_to_cone_boundary_camera(v, obs)
                if cand is not None:
                    candidates.append(cand)

            # 2) Conservative options: slightly slow down, or full stop if inside bubble
            candidates.append(0.7 * v)
            candidates.append(np.zeros(3))

            # Choose candidate that (i) reduces total violations and (ii) minimizes dynamics-aware cost
            best = None
            best_cost = float('inf')
            best_viol_count = 1e9

            for c_camera in candidates:
                # Count violations in camera frame
                c_viol = self._violations_camera(c_camera, obstacles)
                
                # Transform candidate to body frame for dynamics cost
                c_body = self.R_cb @ c_camera
                cost = self._dyn_cost_body(c_body, base_des_body, W_body)

                # Prefer fewer violations, then lower body-frame cost
                if (len(c_viol) < best_viol_count) or (len(c_viol) == best_viol_count and cost < best_cost):
                    best = c_camera
                    best_cost = cost
                    best_viol_count = len(c_viol)

            if best is None:
                break
            v = best

        return v

    def _violations_camera(self, v, obstacles):
        """Check violations in CAMERA frame."""
        viols = []
        v_norm = np.linalg.norm(v) + 1e-9
        ang = None # Initialize ang
        theta = None # Initialize theta
        for obs in obstacles:
            p = obs['p']  # Already in camera frame, relative to camera
            d = np.linalg.norm(p)
            R = obs['r'] + self.safety_margin

            if d <= R:
                # Already inside a safety bubble -> treat as violation (prefer stop)
                rospy.loginfo(f"Violation: Inside safety bubble (d={d:.2f}, R={R:.2f})")
                viols.append(obs)
                continue

            p_hat = p / d
            closing = float(np.dot(v, p_hat))  # component toward obstacle
            if closing <= 0.0:
                continue

            ttc = (d - R) / max(closing, 1e-6)
            if ttc <= self.time_horizon:
                # Also ensure the cone condition (angle smaller than arcsin(R/d))
                theta = math.asin(min(1.0, R / d))
                ang = math.acos(np.clip(np.dot(v / v_norm, p_hat), -1.0, 1.0))
                if ang <= theta:
                    rospy.loginfo(f"Violation: TTC (d={d:.2f}, R={R:.2f}, ttc={ttc:.2f}, time_horizon={self.time_horizon:.2f}, ang={math.degrees(ang):.2f}, theta={math.degrees(theta):.2f})")
                    viols.append(obs)
            else:
                log_ang = f"ang={math.degrees(ang):.2f}" if ang is not None else "ang=N/A"
                log_theta = f"theta={math.degrees(theta):.2f}" if theta is not None else "theta=N/A"
                rospy.loginfo(f"No Violation: d={d:.2f}, R={R:.2f}, ttc={ttc:.2f}, time_horizon={self.time_horizon:.2f}, {log_ang}, {log_theta}")
        rospy.loginfo(f"_violations: Found {len(viols)} violations.")
        return viols

    def _rotate_to_cone_boundary_camera(self, v, obs):
        """Rotate v minimally in CAMERA frame."""
        p = obs['p']  # Already in camera frame
        d = np.linalg.norm(p)
        R = obs['r'] + self.safety_margin
        if d <= R or np.linalg.norm(v) < 1e-6:
            # rospy.loginfo("collision almost surely happened")
            return None

        p_hat = p / d
        speed = np.linalg.norm(v)
        v_hat = v / speed
        closing = np.dot(v, p_hat)
        if closing <= 0.0:
            return v  # already moving away

        theta = math.asin(min(1.0, R / d))
        ang = math.acos(np.clip(np.dot(v_hat, p_hat), -1.0, 1.0))

        if ang > theta + 1e-6:
            return v  # already outside

        # Rotation axis: normal to the plane spanned by {v, p}
        axis = np.cross(v, p)
        na = np.linalg.norm(axis)
        if na < 1e-9:
            # v ~ collinear with p: pick any axis orthogonal to p_hat
            t = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(t, p_hat)) > 0.9:
                t = np.array([0.0, 1.0, 0.0])
            axis = np.cross(p_hat, t)
            na = np.linalg.norm(axis)
            if na < 1e-9:
                return None
        axis /= na

        delta = (theta - ang) + self.boundary_eps  # small epsilon to sit just outside
        v_rot = self._rodrigues(v, axis, delta)
        # Keep original speed
        v_rot = (v_rot / (np.linalg.norm(v_rot) + 1e-9)) * speed
        return v_rot

    def _rodrigues(self, vec, k_hat, angle):
        v = vec
        k = k_hat
        c, s = math.cos(angle), math.sin(angle)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)

    # ---------- Dynamics-aware metric & speed limits ----------

    def _dyn_cost_body(self, v_body, v_des_body, W_body):
        """Compute dynamics cost in BODY frame for proper UAV dynamics."""
        dv = (v_body - v_des_body).reshape(3, 1)
        return float(dv.T.dot(W_body).dot(dv))

    def _clip_speed(self, v):
        """Clip total velocity magnitude while preserving direction."""
        v = np.array(v, dtype=float)
        speed = np.linalg.norm(v)
        
        if speed > self.max_speed:
            # Scale down to max speed while preserving direction
            v = v * (self.max_speed / speed)
        
        return v

    # ---------- Point cloud -> obstacle spheres ----------

    def _spheres_from_cloud_camera(self, pts, k=16):
        """Pick k nearest points in CAMERA frame."""
        if pts is None or pts.shape[0] == 0:
            return []
        # Points already relative to camera
        rel = pts
        d2 = np.sum(rel * rel, axis=1)
        # Filter & take nearest k
        mask = (d2 >= self.min_range**2) & (d2 <= self.max_range**2)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []

        # kth nearest
        take = min(k, idx.size)
        nearest_idx = idx[np.argpartition(d2[idx], take - 1)[:take]]

        obstacles = []
        for i in nearest_idx:
            obstacles.append({'p': pts[i].astype(float), 'r': float(self.point_obstacle_radius)})
        return obstacles


if __name__ == '__main__':
    try:
        node = DynamicsAwareVOFilter3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
