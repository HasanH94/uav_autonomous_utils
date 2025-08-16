#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-only, vegetation-aware + FoV-aware local planner (ROS1, Python)
- Single ROS node with modular classes for easier debugging.
- Publishes velocity setpoints at high rate; optional short-horizon path (preview toggle ready).
- Multi-constraint CBF-QP (OSQP) with:
  * Obstacle CBF from live point cloud (nearest supports), Δt-aware safety margin
  * Vegetation-aware weighting (temporal persistence + variance → kappa)
  * Optional FoV-CBF (keep nest in camera FoV) with finite-difference row
  * Velocity and acceleration (slew-rate) box constraints

Assumptions:
- ROS1 (Noetic) with mavros
- Topics:
  * /camera/depth/points : sensor_msgs/PointCloud2
  * /uav/odometry        : nav_msgs/Odometry (ENU, frame ~ "odom")
  * /move_base_simple/goal: geometry_msgs/PoseStamped (goal pose)
  * /apf_vel             : geometry_msgs/TwistStamped (nominal linear velocity) — optional
- Outputs:
  * /mavros/setpoint_velocity/cmd_vel : geometry_msgs/TwistStamped
  * /cbf_local_planner/path           : nav_msgs/Path (low-rate viz)

Notes:
- Install deps: pip3 install osqp scipy numpy
- Keep parameters small initially; this is a working scaffold to iterate on hardware quickly.
"""

import math
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool

# Try importing OSQP + SciPy; fail gracefully with a message
try:
    import osqp
    import scipy.sparse as sp
except ImportError as e:
    osqp = None
    sp = None
    rospy.logwarn("OSQP or SciPy not found. Install with: pip3 install osqp scipy. QP will not run.")

# ----------------------------- Utilities -----------------------------

def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

@dataclass
class SupportPoint:
    p: np.ndarray   # 3x, position of nearest support point (odom)
    g: np.ndarray   # 3x, outward unit normal from support to UAV (odom)
    d: float        # distance ||q - p||
    kappa: float    # confidence [0,1] (rigid=1 ←→ leafy=0)

# ------------------------ Cloud ring buffer (odom) ------------------------
class CloudRingBuffer:
    """Maintains last N point clouds transformed into odom frame for temporal tests.
       Keeps decimated arrays (Nx3 float32). Lightweight, no KD-tree to keep code simple (OK if we decimate strongly).
    """
    def __init__(self, max_len=8, crop_radius=5.0, stride=4):
        self.max_len = max_len
        self.crop_r2 = crop_radius * crop_radius
        self.stride = max(1, int(stride))
        self.lock = threading.Lock()
        self.buff: List[np.ndarray] = []   # list of (M_i x 3)
        self.frames: List[float] = []      # timestamps

    def add_cloud(self, cloud_msg: PointCloud2, tf_listener: tf.TransformListener, odom_frame: str) -> Optional[float]:
        # Convert to Nx3, decimate by stride and crop radius, then transform to odom
        if cloud_msg is None:
            return None
        # Extract points in camera frame
        pts = []
        stride = self.stride
        r2max = self.crop_r2
        for i, p in enumerate(pc2.read_points(cloud_msg, field_names=("x","y","z"), skip_nans=True)):
            if i % stride != 0:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if (x*x + y*y + z*z) > r2max:
                continue
            pts.append((x,y,z))
        if not pts:
            return None
        pts = np.asarray(pts, dtype=np.float32)

        # Transform to odom
        try:
            tf_listener.waitForTransform(odom_frame, cloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(0.02))
            (trans, rot) = tf_listener.lookupTransform(odom_frame, cloud_msg.header.frame_id, rospy.Time(0))
            R = tft.quaternion_matrix(rot)[:3,:3]
            t = np.array(trans).reshape(1,3)
            pts_odom = (pts @ R.T) + t
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

        with self.lock:
            self.buff.append(pts_odom.astype(np.float32))
            self.frames.append(cloud_msg.header.stamp.to_sec())
            if len(self.buff) > self.max_len:
                self.buff.pop(0)
                self.frames.pop(0)
        return self.frames[-1]

    def nearest_supports(self, q_odom: np.ndarray, k: int = 1, region_radius: float = 0.25,
                          min_neighbors: int = 12) -> List[SupportPoint]:
        """Return up to k support points near q with kappa from temporal persistence+variance.
        - region_radius: radius around nearest point to evaluate variance/persistence
        """
        with self.lock:
            if not self.buff:
                return []
            # Use latest cloud to find nearest point (linear scan; OK when decimated)
            cloud = self.buff[-1]

        diffs = cloud - q_odom.reshape(1,3)
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        if d2.size == 0:
            return []
        idx_sorted = np.argsort(d2)[:max(k, min_neighbors)]

        supports: List[SupportPoint] = []
        chosen = 0
        for idx in idx_sorted:
            p = cloud[idx]
            d = float(math.sqrt(max(1e-12, d2[idx])))
            if d < 1e-6:
                continue
            g = (q_odom - p) / d  # outward normal in odom

            # Neighborhood in current cloud
            nbr_mask = np.linalg.norm(cloud - p.reshape(1,3), axis=1) <= region_radius
            nbrs = cloud[nbr_mask]
            if nbrs.shape[0] < min_neighbors:
                continue

            # PCA for surface variation (low = planar/rigid-ish, high = fluffy)
            mu = np.mean(nbrs, axis=0)
            X = nbrs - mu
            C = (X.T @ X) / max(1, X.shape[0]-1)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 0.0, None)
            # Surface variation metric (smallest eigenvalue / sum)
            surf_var = float(evals[0] / max(1e-9, np.sum(evals)))

            # Temporal persistence + variance along g (look back over buffer)
            hits = 0; total = 0; proj_vals = []
            with self.lock:
                for pc_arr in self.buff[:-1]:  # exclude current already used
                    total += 1
                    mask = np.linalg.norm(pc_arr - p.reshape(1,3), axis=1) <= region_radius
                    if np.count_nonzero(mask) > 0:
                        hits += 1
                        proj = (pc_arr[mask] - p.reshape(1,3)) @ g.reshape(3,1)
                        proj_vals.append(proj.flatten())
            persistence = hits / max(1, total)
            if proj_vals:
                proj_concat = np.concatenate(proj_vals)
                var_along_g = float(np.var(proj_concat))
            else:
                var_along_g = 0.05  # if no history, assume moderate variance

            # Confidence kappa in [0,1]
            # Tunables
            tau = 0.5         # persistence threshold
            a = 6.0           # slope of sigmoid
            sigma0 = 0.09     # m^2 variance scale (~30 cm std → down-weight)
            pers_term = 1.0 / (1.0 + math.exp(-a*(persistence - tau)))
            var_term  = math.exp(-var_along_g / max(1e-6, sigma0))
            kappa = pers_term * var_term * (1.0 - clip(surf_var*3.0, 0.0, 1.0))
            kappa = float(clip(kappa, 0.0, 1.0))

            supports.append(SupportPoint(p=p.astype(np.float64), g=g.astype(np.float64), d=d, kappa=kappa))
            chosen += 1
            if chosen >= k:
                break

        return supports

# ------------------------ FoV constraint helper ------------------------
class FoVConstraint:
    def __init__(self, tf_listener: tf.TransformListener, odom_frame: str, camera_frame: str,
                 theta_deg: float = 40.0):
        self.tf_listener = tf_listener
        self.odom_frame = odom_frame
        self.camera_frame = camera_frame
        self.cos_theta = math.cos(math.radians(theta_deg))
        # Optical frame convention: +Z forward
        self.c_optical = np.array([0.0, 0.0, 1.0])

    def compute_h_and_row(self, nest_world: np.ndarray, uav_pose_world: np.ndarray,
                          xi: np.ndarray, dt_fd: float = 0.05) -> Optional[Tuple[float, np.ndarray]]:
        """Return (h_fov, a_fov[4]) using finite-difference sensitivity wrt [ux,uy,uz,omega_z].
        - nest_world: 3x target position in odom
        - uav_pose_world: 4x4 pose matrix of UAV body (odom)
        - xi: current decision vector [ux,uy,uz,omega_z]
        """
        try:
            self.tf_listener.waitForTransform(self.odom_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.02))
            (tc, qc) = self.tf_listener.lookupTransform(self.odom_frame, self.camera_frame, rospy.Time(0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

        # Camera axis c in odom
        Rc = tft.quaternion_matrix(qc)[:3,:3]
        c_world = (Rc @ self.c_optical.reshape(3,1)).reshape(3)
        p_cam = np.array(tc).reshape(3)

        r = nest_world - p_cam
        nr = np.linalg.norm(r)
        if nr < 1e-6:
            return None
        r = r / nr
        h0 = self.cos_theta - float(np.dot(c_world, r))

        # Finite diff for row a = dh/d[xi]
        a = np.zeros(4, dtype=np.float64)

        def step_h(dxi):
            # Propagate camera pose by small step using UAV world-frame [u, omega_z]
            u = dxi[:3] * dt_fd
            dyaw = dxi[3] * dt_fd
            # Update camera position (translate by u)
            p_cam_s = p_cam + u
            # Update camera axis by yaw about world z
            Rz = tft.rotation_matrix(dyaw, (0,0,1))[:3,:3]
            c_world_s = (Rz @ c_world.reshape(3,1)).reshape(3)
            r_s = nest_world - p_cam_s
            nr_s = np.linalg.norm(r_s)
            if nr_s < 1e-6:
                return h0
            r_s = r_s / nr_s
            return self.cos_theta - float(np.dot(c_world_s, r_s))

        eps = 0.1  # small finite diff in (m/s) and (rad/s)
        for j in range(4):
            dxi = np.zeros(4)
            dxi[j] = eps
            a[j] = (step_h(xi + dxi) - h0) / eps

        return (h0, a)

# ------------------------ CBF-QP core ------------------------
class CBF_QP:
    def __init__(self, v_max=2.5, omega_max=2.0, a_max=3.0, alpha_min=2.0, alpha_max=8.0,
                 d_min=0.30, d_extra=0.18):
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.d_min = d_min
        self.d_extra = d_extra
        self.x_prev = np.zeros(4)  # [ux, uy, uz, wz] last commanded

        if osqp is None or sp is None:
            rospy.logwarn("OSQP/SciPy not available. QP will be skipped.")

    def assemble_and_solve(self, xi0: np.ndarray, supports: List[SupportPoint], h_fov_row=None,
                           dt: float = 0.02, dt_sense: float = 0.03, dt_hold: float = 0.02,
                           lambda_clf: float = 0.0, clf_vec: Optional[np.ndarray] = None,
                           weights_soft=(10.0, 1e4)) -> np.ndarray:
        """Build small QP with per-row slack and solve via OSQP. Returns xi (4,).
        - xi0: nominal [ux,uy,uz,wz]
        - supports: up to K SupportPoint entries
        - h_fov_row: optional tuple (h_fov, a_fov)
        - dt: controller tick
        - dt_sense: time since last depth frame
        - dt_hold: worst-case command hold (offboard)
        - lambda_clf: optional weight on a CLF quadratic (use clf_vec as target direction)
        - weights_soft: (w_soft, w_hard) for slack penalty depending on kappa
        """
        x_prev = self.x_prev.copy()
        K = len(supports)
        n = 4 + K  # decision vars = [ux,uy,uz,wz, s1..sK]

        # Cost: 0.5*(x-x0)^T I (x-x0) + sum w_i s_i^2 + optional CLF term
        # P (quadratic) and q (linear) in OSQP form: 0.5 x^T P x + q^T x
        if osqp is None or sp is None:
            # Fallback: just saturate xi0 and return
            xi = xi0.copy()
            xi[:3] = np.clip(xi[:3], -self.v_max, self.v_max)
            xi[3] = clip(xi[3], -self.omega_max, self.omega_max)
            self.x_prev = xi
            return xi

        P_diag = np.ones(4)
        # Add slack weights
        w_soft, w_hard = weights_soft
        slack_weights = []
        for spt in supports:
            w_i = w_soft + spt.kappa * (w_hard - w_soft)
            slack_weights.append(w_i)
        if K > 0:
            P_diag = np.concatenate([P_diag, np.array(slack_weights)])
        P = sp.diags(P_diag)

        # Linear term encourages staying near xi0
        q = np.zeros(n)
        q[:4] = -xi0  # 0.5||x - x0||^2 -> x^T x/2 - x0^T x + const → q = -x0

        # Optional CLF: pull toward clf_vec (unit dir), small weight into P (diagonalized approx)
        # For simplicity, add lambda to the linear term toward clf_vec (regularizer): q[:3] += -lambda*clf_vec
        if lambda_clf > 0.0 and clf_vec is not None:
            q[:3] += -lambda_clf * clf_vec

        rows_A = []
        l = []
        u = []

        # 1) Obstacle CBF rows with Δt-aware safety and slack per row
        for i, spt in enumerate(supports):
            # Effective safety margin
            d_safe = self.d_min + spt.kappa * self.d_extra
            d_safe_eff = d_safe + np.linalg.norm(x_prev[:3]) * dt_sense + 0.5 * self.a_max * (dt_hold**2)
            h = spt.d - d_safe_eff
            alpha_i = self.alpha_min + spt.kappa * (self.alpha_max - self.alpha_min)

            # Row: [g^T, 0, ..., -1_i, ..., 0]
            a = np.zeros(n)
            a[0:3] = spt.g
            # yaw doesn't affect distance instantaneously → coefficient 0 on omega_z
            a[4 + i] = -1.0
            rows_A.append(a)
            l.append(-alpha_i * h)  # a^T x >= -alpha*h  → lower bound
            u.append(np.inf)        # no upper bound

        # 2) FoV-CBF row (optional, no slack here)
        if h_fov_row is not None:
            h_fov, a_fov = h_fov_row
            a = np.zeros(n)
            a[0:4] = a_fov
            rows_A.append(a)
            l.append(-4.0 * h_fov)  # alpha_fov ~ 4; make param if needed
            u.append(np.inf)

        # 3) Velocity box: |u_k| <= v_max ; |omega| <= omega_max
        # as l <= A x <= u
        for j in range(3):
            a = np.zeros(n); a[j] = 1.0
            rows_A.append(a); l.append(-self.v_max); u.append(self.v_max)
        a = np.zeros(n); a[3] = 1.0
        rows_A.append(a); l.append(-self.omega_max); u.append(self.omega_max)

        # 4) Accel (slew-rate) box: |u_k - u_prev_k| <= a_max * dt ; same for omega
        for j in range(3):
            a = np.zeros(n); a[j] = 1.0
            rows_A.append(a); l.append(x_prev[j] - self.a_max*dt); u.append(x_prev[j] + self.a_max*dt)
        a = np.zeros(n); a[3] = 1.0
        rows_A.append(a); l.append(x_prev[3] - self.a_max*dt); u.append(x_prev[3] + self.a_max*dt)

        # Stack
        A = sp.csc_matrix(np.vstack(rows_A))
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)

        # Solve with OSQP
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True, warm_start=True)
        res = prob.solve()

        if res.info.status_val not in (1, 2):
            rospy.logwarn_throttle(1.0, f"OSQP status: {res.info.status}; falling back to xi0")
            xi = xi0.copy()
        else:
            x = np.array(res.x).reshape(-1)
            xi = x[:4]

        # Update memory
        self.x_prev = xi.copy()
        return xi

# ------------------------ Main ROS node ------------------------
class DepthCBFLocalPlanner:
    def __init__(self):
        rospy.init_node('depth_cbf_local_planner')
        # Params
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_depth_optical_frame')
        self.max_speed = rospy.get_param('~max_speed', 2.5)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 2.0)
        self.a_max = rospy.get_param('~a_max', 3.0)
        self.yaw_switch_distance = rospy.get_param('~yaw_switch_distance', 1.0)
        self.enable_fov = rospy.get_param('~enable_fov', True)
        self.fov_half_angle_deg = rospy.get_param('~fov_half_angle_deg', 40.0)
        self.enable_preview = rospy.get_param('~enable_preview', False)
        self.preview_horizon_s = rospy.get_param('~preview_horizon_s', 1.2)
        self.preview_points = rospy.get_param('~preview_points', 4)
        self.pub_path_rate = rospy.get_param('~path_rate_hz', 10.0)
        self.support_k = rospy.get_param('~support_k', 1)

        # Vegetation-aware params
        self.region_radius = rospy.get_param('~region_radius', 0.20)  # m neighborhood for kappa
        self.min_neighbors = rospy.get_param('~min_neighbors', 12)

        # Δt-aware sampled-data params
        self.dt_hold_default = rospy.get_param('~dt_hold_default', 0.02)  # 50 Hz hold assumption

        # State
        self.goal = None
        self.goal_orientation = None
        self.current_pose = None  # np.array([x,y,z])
        self.current_yaw = 0.0
        self.apf_linear_velocity = np.zeros(3)
        self.last_depth_stamp = None

        self.tf_listener = tf.TransformListener()
        self.cloud_buf = CloudRingBuffer(max_len=8, crop_radius=5.0, stride=4)
        self.fov_helper = FoVConstraint(self.tf_listener, self.odom_frame, self.camera_frame, self.fov_half_angle_deg)
        self.qp = CBF_QP(v_max=self.max_speed, omega_max=self.max_yaw_rate, a_max=self.a_max)

        # ROS I/O
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb)
        rospy.Subscriber('/apf_vel', TwistStamped, self.apf_vel_cb)
        self.pub_cmd = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        self.pub_path = rospy.Publisher('/cbf_local_planner/path', Path, queue_size=1)

        self.rate = rospy.Rate(50)  # main loop
        self.path_timer = rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.pub_path_rate)), self.publish_path_cb)

    # ---------------- Callbacks ----------------
    def cloud_cb(self, msg: PointCloud2):
        t = self.cloud_buf.add_cloud(msg, self.tf_listener, self.odom_frame)
        if t is not None:
            self.last_depth_stamp = t

    def goal_cb(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        self.goal_orientation = msg.pose.orientation

    def apf_vel_cb(self, msg: TwistStamped):
        self.apf_linear_velocity[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]

    def odom_cb(self, msg: Odometry):
        self.current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float64)
        q = msg.pose.pose.orientation
        (_, _, yaw) = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    # ---------------- Helper methods ----------------
    def hybrid_desired_yaw(self) -> float:
        if self.goal is None or self.current_pose is None:
            return self.current_yaw
        horizontal_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
        desired_yaw = self.current_yaw
        if horizontal_dist > self.yaw_switch_distance:
            dx = self.goal[0] - self.current_pose[0]
            dy = self.goal[1] - self.current_pose[1]
            if abs(dx) > 1e-2 or abs(dy) > 1e-2:
                desired_yaw = math.atan2(dy, dx)
        elif self.goal_orientation is not None:
            qg = [self.goal_orientation.x, self.goal_orientation.y, self.goal_orientation.z, self.goal_orientation.w]
            (_, _, desired_yaw) = tft.euler_from_quaternion(qg)
        return desired_yaw

    def compute_nominal(self, dt: float) -> np.ndarray:
        # Nominal xi0 = [ux,uy,uz,omega_z]: use APF linear velocity + hybrid yaw PID (simple P-D here)
        xi0 = np.zeros(4)
        # Linear part from /apf_vel (already limited later)
        xi0[:3] = self.apf_linear_velocity.copy()
        # Yaw part from hybrid strategy (PD)
        psi_d = self.hybrid_desired_yaw()
        e = math.atan2(math.sin(psi_d - self.current_yaw), math.cos(psi_d - self.current_yaw))
        kp_yaw = rospy.get_param('~kp_yaw', 0.8)
        kd_yaw = rospy.get_param('~kd_yaw', 0.1)
        # Derivative not tracked → approximate with 0 for simplicity; you can wire rate from odom
        xi0[3] = clip(kp_yaw * e, -self.max_yaw_rate, self.max_yaw_rate)
        return xi0

    def maybe_fov_row(self, xi: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        if not self.enable_fov or self.goal is None:
            return None
        # Construct UAV pose 4x4 (only yaw used in helper)
        T = np.eye(4)
        T[:3,3] = self.current_pose if self.current_pose is not None else np.zeros(3)
        T[:3,:3] = tft.euler_matrix(0,0,self.current_yaw)[:3,:3]
        return self.fov_helper.compute_h_and_row(self.goal, T, xi)

    # ---------------- Path publisher (viz) ----------------
    def publish_path_cb(self, event):
        if self.current_pose is None:
            return
        # Simple integration of last commanded (qp.x_prev) for visualization only
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.odom_frame
        dt = 1.0 / max(1e-3, self.pub_path_rate)
        pos = self.current_pose.copy()
        yaw = self.current_yaw
        xi = self.qp.x_prev.copy()
        for i in range(max(2, int(self.preview_horizon_s / dt))):
            pos = pos + xi[:3] * dt
            yaw = yaw + xi[3] * dt
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = pos[0]
            ps.pose.position.y = pos[1]
            ps.pose.position.z = pos[2]
            q = tft.quaternion_from_euler(0,0,yaw)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
            path.poses.append(ps)
        self.pub_path.publish(path)

    # ---------------- Main loop ----------------
    def spin(self):
        last = time.time()
        while not rospy.is_shutdown():
            now = time.time(); dt = now - last; last = now
            if dt <= 0.0:
                self.rate.sleep(); continue
            if self.goal is None or self.current_pose is None:
                self.rate.sleep(); continue

            xi0 = self.compute_nominal(dt)
            # Supports
            supports = self.cloud_buf.nearest_supports(self.current_pose, k=self.support_k,
                                                        region_radius=self.region_radius,
                                                        min_neighbors=self.min_neighbors)
            # Δt-aware terms
            if self.last_depth_stamp is None:
                dt_sense = 0.05
            else:
                dt_sense = max(0.0, rospy.Time.now().to_sec() - self.last_depth_stamp)
            dt_hold = self.dt_hold_default

            # FoV row (optional)
            fov_row = self.maybe_fov_row(xi0)

            # CLF direction to 1.2m standoff (optional small pull)
            standoff = 1.20
            v_clf = None
            if self.goal is not None and self.current_pose is not None:
                vec = self.goal - self.current_pose
                n = np.linalg.norm(vec)
                if n > 1e-3:
                    p_stand = self.goal - standoff * (vec / n)
                    v_dir = p_stand - self.current_pose
                    nv = np.linalg.norm(v_dir)
                    if nv > 1e-3:
                        v_clf = v_dir / nv

            xi = self.qp.assemble_and_solve(xi0, supports, h_fov_row=fov_row, dt=dt, dt_sense=dt_sense,
                                             dt_hold=dt_hold, lambda_clf=0.1 if v_clf is not None else 0.0,
                                             clf_vec=v_clf)

            # Publish velocity setpoint
            tw = TwistStamped()
            tw.header.stamp = rospy.Time.now()
            tw.header.frame_id = self.odom_frame
            tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = xi[0], xi[1], xi[2]
            tw.twist.angular.z = xi[3]
            self.pub_cmd.publish(tw)

            # (Optional) preview trajectory streaming can be added as a separate timer using MAVROS' SetpointTrajectory
            # — kept off by default for max performance. When enabled, generate MultiDOFJointTrajectory here.

            self.rate.sleep()

# ------------------------ Entrypoint ------------------------
if __name__ == '__main__':
    try:
        node = DepthCBFLocalPlanner()
        node.spin()
    except rospy.ROSInterruptException:
        pass
