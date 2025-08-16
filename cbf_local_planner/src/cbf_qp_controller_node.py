#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
from typing import Optional, Tuple

import numpy as np
import rospy
import tf
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from uav_autonomous_utils.msg import SupportArray

try:
    import osqp
    import scipy.sparse as sp
except Exception:
    osqp = None
    sp = None


def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class FoVConstraint:
    def __init__(self, tf_listener: tf.TransformListener, odom_frame: str, camera_frame: str, theta_deg: float):
        self.tf_listener = tf_listener
        self.odom_frame = odom_frame
        self.camera_frame = camera_frame
        self.cos_theta = math.cos(math.radians(theta_deg))
        self.c_optical = np.array([0.0, 0.0, 1.0])  # optical +Z forward

    def compute(self, nest_world: np.ndarray, xi: np.ndarray, dt_fd: float = 0.05) -> Optional[Tuple[float, np.ndarray]]:
        try:
            self.tf_listener.waitForTransform(self.odom_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.02))
            (tc, qc) = self.tf_listener.lookupTransform(self.odom_frame, self.camera_frame, rospy.Time(0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
        Rc = tft.quaternion_matrix(qc)[:3,:3]
        c_world = (Rc @ self.c_optical.reshape(3,1)).reshape(3)
        p_cam = np.array(tc).reshape(3)

        r = nest_world - p_cam
        nr = np.linalg.norm(r)
        if nr < 1e-6:
            return None
        r = r / max(nr, 1e-6)
        h0 = self.cos_theta - float(np.dot(c_world, r))

        a = np.zeros(4)
        def step_h(dxi):
            u = dxi[:3] * dt_fd
            dyaw = dxi[3] * dt_fd
            p_cam_s = p_cam + u
            Rz = tft.rotation_matrix(dyaw, (0,0,1))[:3,:3]
            c_world_s = (Rz @ c_world.reshape(3,1)).reshape(3)
            r_s = nest_world - p_cam_s
            nr_s = np.linalg.norm(r_s)
            if nr_s < 1e-6:
                return h0
            r_s = r_s / max(nr_s, 1e-6)
            return self.cos_theta - float(np.dot(c_world_s, r_s))
        eps = 0.1
        for j in range(4):
            dxi = np.zeros(4); dxi[j] = eps
            a[j] = (step_h(xi + dxi) - h0) / eps
        return (h0, a)


class QPController:
    def __init__(self, v_max, omega_max, a_max, alpha_min, alpha_max, d_min, d_extra):
        self.v_max = v_max
        self.omega_max = omega_max
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.d_min = d_min
        self.d_extra = d_extra
        self.x_prev = np.zeros(4)

    def solve(self, xi0: np.ndarray, supports, h_fov_row, dt: float, dt_sense: float, dt_hold: float,
              lambda_clf: float = 0.0, clf_vec: Optional[np.ndarray] = None):
        x_prev = self.x_prev.copy()
        K = len(supports)
        n = 4 + K
        if osqp is None or sp is None:
            xi = xi0.copy()
            xi[:3] = np.clip(xi[:3], -self.v_max, self.v_max)
            xi[3] = clip(xi[3], -self.omega_max, self.omega_max)
            self.x_prev = xi
            return xi

        # Quadratic cost: 0.5||(x - x0)||^2 + sum w_i s_i^2
        P_diag = np.ones(4)
        w_soft, w_hard = 10.0, 1e4
        slack_weights = [w_soft + sp_i.kappa * (w_hard - w_soft) for sp_i in supports]
        if K > 0:
            P_diag = np.concatenate([P_diag, np.array(slack_weights)])
        P = sp.diags(P_diag)
        q = np.zeros(n); q[:4] = -xi0
        if lambda_clf > 0.0 and clf_vec is not None:
            q[:3] += -lambda_clf * clf_vec

        rows = []; lo = []; up = []
        # Obstacle rows with Δt-aware safety, per-row slack
        for i, sp_i in enumerate(supports):
            d_safe = self.d_min + sp_i.kappa * self.d_extra
            d_eff = d_safe + np.linalg.norm(x_prev[:3]) * dt_sense + 0.5 * self.a_max * (dt_hold**2)
            h = sp_i.d - d_eff
            alpha_i = self.alpha_min + sp_i.kappa * (self.alpha_max - self.alpha_min)
            a = np.zeros(n); a[0:3] = [sp_i.g.x, sp_i.g.y, sp_i.g.z]; a[4+i] = -1.0
            rows.append(a); lo.append(-alpha_i * h); up.append(np.inf)

        # FoV row (optional)
        if h_fov_row is not None:
            h_fov, a_fov = h_fov_row
            a = np.zeros(n); a[0:4] = a_fov
            rows.append(a); lo.append(-4.0 * h_fov); up.append(np.inf)  # alpha_fov ~ 4

        # Velocity box
        for j in range(3):
            a = np.zeros(n); a[j] = 1.0
            rows.append(a); lo.append(-self.v_max); up.append(self.v_max)
        a = np.zeros(n); a[3] = 1.0
        rows.append(a); lo.append(-self.omega_max); up.append(self.omega_max)

        # Accel (slew-rate) box
        for j in range(3):
            a = np.zeros(n); a[j] = 1.0
            rows.append(a); lo.append(x_prev[j] - self.a_max*dt); up.append(x_prev[j] + self.a_max*dt)
        a = np.zeros(n); a[3] = 1.0
        rows.append(a); lo.append(x_prev[3] - self.a_max*dt); up.append(x_prev[3] + self.a_max*dt)

        A = sp.csc_matrix(np.vstack(rows))
        lo = np.array(lo, dtype=np.float64); up = np.array(up, dtype=np.float64)
        prob = osqp.OSQP(); prob.setup(P=P, q=q, A=A, l=lo, u=up, verbose=False, polish=True, warm_start=True)
        res = prob.solve()
        if res.info.status_val not in (1, 2):
            rospy.logwarn_throttle(1.0, f"OSQP status: {res.info.status}; falling back to xi0")
            xi = xi0.copy()
        else:
            xi = np.array(res.x).reshape(-1)[:4]
        self.x_prev = xi.copy()
        return xi


class ControllerNode:
    def __init__(self):
        rospy.init_node('cbf_qp_controller_node')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_depth_optical_frame')
        self.max_speed = rospy.get_param('~max_speed', 2.5)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 2.0)
        self.a_max = rospy.get_param('~a_max', 3.0)
        self.enable_fov = rospy.get_param('~enable_fov', True)
        self.fov_half_angle_deg = rospy.get_param('~fov_half_angle_deg', 40.0)
        self.yaw_switch_distance = rospy.get_param('~yaw_switch_distance', 1.0)
        self.dt_hold_default = rospy.get_param('~dt_hold_default', 0.02)
        self.support_k = rospy.get_param('~support_k', 1)
        self.publish_cmd_vel = rospy.get_param('~publish_cmd_vel', True)
        
        # Goal acceptance parameters
        self.goal_pos_tol = rospy.get_param('~goal_pos_tol', 0.10)
        self.goal_yaw_tol = rospy.get_param('~goal_yaw_tol', 0.10)
        self.goal_speed_tol = rospy.get_param('~goal_speed_tol', 0.15)
        self.goal_hold_time = rospy.get_param('~goal_hold_time', 0.3)
        self.require_fov_at_goal = rospy.get_param('~require_fov_at_goal', False)
        self.freeze_cmd_on_goal = rospy.get_param('~freeze_cmd_on_goal', False)

        self.tf_listener = tf.TransformListener()
        self.fov = FoVConstraint(self.tf_listener, self.odom_frame, self.camera_frame, self.fov_half_angle_deg)
        self.qp = QPController(self.max_speed, self.max_yaw_rate, self.a_max, 2.0, 8.0, 0.30, 0.18)

        rospy.Subscriber('/cbf/supports', SupportArray, self.supports_cb, queue_size=1)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/nominal_vel', TwistStamped, self.apf_cb)

        self.pub_cmd = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        self.pub_path = rospy.Publisher('/cbf/controller_path', Path, queue_size=1)
        self.pub_cmd_echo = rospy.Publisher('/cbf/last_cmd', TwistStamped, queue_size=1)
        self.pub_goal_reached = rospy.Publisher('/cbf/goal_reached', Bool, queue_size=1, latch=True)

        self.goal = None; self.goal_q = None
        self.pose = None; self.yaw = 0.0
        self.apf = np.zeros(3)
        self.last_depth_stamp = None
        self._in_goal_since = None
        self._goal_reached = False

        self.rate = rospy.Rate(50)

    def supports_cb(self, msg: SupportArray):
        self.last_depth_stamp = msg.header.stamp.to_sec()
        self.supports = msg.supports

    def odom_cb(self, msg: Odometry):
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float64)
        q = msg.pose.pose.orientation
        (_,_,self.yaw) = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])

    def goal_cb(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        self.goal_q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        # Reset goal reached status on new goal
        self._in_goal_since = None
        self._goal_reached = False
        self.pub_goal_reached.publish(Bool(False))

    def apf_cb(self, msg: TwistStamped):
        self.apf[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]

    def hybrid_desired_yaw(self) -> float:
        if self.goal is None or self.pose is None:
            return self.yaw
        horizontal_dist = np.linalg.norm(self.goal[:2] - self.pose[:2])
        desired = self.yaw
        if horizontal_dist > self.yaw_switch_distance:
            dx = self.goal[0] - self.pose[0]; dy = self.goal[1] - self.pose[1]
            if abs(dx) > 1e-2 or abs(dy) > 1e-2:
                desired = math.atan2(dy, dx)
        elif self.goal_q is not None:
            (_,_,desired) = tft.euler_from_quaternion(self.goal_q)
        return desired

    def maybe_fov_row(self, xi) -> Optional[Tuple[float, np.ndarray]]:
        if not self.enable_fov or self.goal is None:
            return None
        return self.fov.compute(self.goal, xi)

    def compute_nominal(self):
        xi0 = np.zeros(4)
        kp_pos = rospy.get_param('~kp_pos', 1.2)  # 1/s
        if np.linalg.norm(self.apf) < 1e-3 and self.goal is not None and self.pose is not None:
            # Goal-seeking when APF is absent or near goal
            xi0[:3] = np.clip(kp_pos * (self.goal - self.pose), -self.max_speed, self.max_speed)
        else:
            xi0[:3] = self.apf.copy()
        psi_d = self.hybrid_desired_yaw()
        e = math.atan2(math.sin(psi_d - self.yaw), math.cos(psi_d - self.yaw))
        kp = rospy.get_param('~kp_yaw', 0.8)
        xi0[3] = clip(kp * e, -self.max_yaw_rate, self.max_yaw_rate)
        return xi0

    def spin(self):
        last = time.time()
        supports = []
        while not rospy.is_shutdown():
            now = time.time(); dt = now - last; last = now
            if self.pose is None or self.goal is None:
                self.rate.sleep(); continue
            xi0 = self.compute_nominal()
            dt_sense = 0.05 if self.last_depth_stamp is None else max(0.0, rospy.Time.now().to_sec() - self.last_depth_stamp)
            dt_hold = self.dt_hold_default

            fov_row = self.maybe_fov_row(xi0)
            # CLF pull toward standoff (0.0 for exact goal)
            standoff = 0.0
            v_clf = None
            vec = self.goal - self.pose
            n = np.linalg.norm(vec)
            if n > 1e-3:
                p_stand = self.goal - standoff * (vec / n)
                v_dir = p_stand - self.pose
                nv = np.linalg.norm(v_dir)
                if nv > 1e-3:
                    v_clf = v_dir / nv
            xi = self.qp.solve(xi0, getattr(self, 'supports', []), fov_row, dt, dt_sense, dt_hold,
                               lambda_clf=0.15 if v_clf is not None else 0.0, clf_vec=v_clf)
            
            # Check goal reached conditions
            pos_err = float(np.linalg.norm(self.goal - self.pose))
            psi_des = self.hybrid_desired_yaw()
            yaw_err = math.atan2(math.sin(psi_des - self.yaw), math.cos(psi_des - self.yaw))
            speed = float(np.linalg.norm(xi[:3]))
            
            fov_ok = True
            if self.require_fov_at_goal:
                eval_row = self.fov.compute(self.goal, xi)
                if eval_row is not None:
                    h_fov, _ = eval_row
                    fov_ok = (h_fov <= 0.02)  # small margin inside cone
                else:
                    fov_ok = False
            
            ok_now = (pos_err <= self.goal_pos_tol) and (abs(yaw_err) <= self.goal_yaw_tol) \
                     and (speed <= self.goal_speed_tol) and fov_ok
            
            t_now = rospy.Time.now().to_sec()
            if ok_now:
                if self._in_goal_since is None:
                    self._in_goal_since = t_now
                if t_now - self._in_goal_since >= self.goal_hold_time:
                    if not self._goal_reached:
                        self._goal_reached = True
                        self.pub_goal_reached.publish(Bool(True))
                        rospy.loginfo("Goal reached!")
            else:
                if self._in_goal_since is not None:
                    self._in_goal_since = None
                if self._goal_reached:
                    self._goal_reached = False
                    self.pub_goal_reached.publish(Bool(False))
            
            # Freeze command if goal reached and freeze_cmd_on_goal is enabled
            if self._goal_reached and self.freeze_cmd_on_goal:
                xi[:] = 0.0

            # Publish cmd_vel (only if enabled)
            tw = TwistStamped(); tw.header.stamp = rospy.Time.now(); tw.header.frame_id = self.odom_frame
            tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = xi[0], xi[1], xi[2]
            tw.twist.angular.z = xi[3]
            if self.publish_cmd_vel:
                self.pub_cmd.publish(tw)
            self.pub_cmd_echo.publish(tw)  # keep echo for the streamer always

            # Publish a tiny path for viz (integrating last command)
            path = Path(); path.header.stamp = tw.header.stamp; path.header.frame_id = self.odom_frame
            pos = self.pose.copy(); yaw = self.yaw; dtp = 0.1
            for _ in range(10):
                pos = pos + xi[:3] * dtp
                yaw = yaw + xi[3] * dtp
                ps = PoseStamped(); ps.header = path.header
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos[0], pos[1], pos[2]
                qy = tft.quaternion_from_euler(0,0,yaw)
                ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qy
                path.poses.append(ps)
            self.pub_path.publish(path)

            self.rate.sleep()


if __name__ == '__main__':
    node = ControllerNode()
    node.spin()