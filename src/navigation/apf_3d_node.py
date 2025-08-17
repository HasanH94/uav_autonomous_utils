#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class APF3DNode(object):
    """
    3D APF (Artificial Potential Field) velocity corrector for drones.

    Subscriptions:
      - ~attractive_cmd (geometry_msgs/TwistStamped): desired linear velocity (xyz)
        and desired yaw rate in angular.z. angular.x/y are ignored.
      - ~pointcloud (sensor_msgs/PointCloud2): obstacle points in the SAME frame
        as the velocity command (typically 'base_link' or body frame).
        We assume the drone is at the origin of this frame.

    Publications:
      - ~cmd (geometry_msgs/TwistStamped): corrected linear velocity (xyz) +
        yaw rate copied from attractive input (angular.z). angular.x/y = 0.

    Repulsive velocity model (Khatib-style):
      For each obstacle point p with distance d = ||p|| < r:
        v_rep contribution = -eta * (1/d - 1/r) * (1/d^3) * p
      Summed over all points in range. This points away from obstacles.
      The resulting vector is optionally low-pass filtered and clipped.

    Parameters (~namespace):
      r               [double, m]   : repulsion influence radius (default: 2.0)
      eta_xy          [double]      : repulsion gain for XY (default: 0.8)
      eta_z           [double]      : repulsion gain for Z (default: 0.8)
      eps             [double, m]   : small distance to avoid 1/0 (default: 0.10)
      max_rep_norm    [double, m/s] : cap magnitude of repulsive velocity (default: 1.0)
      max_out_norm    [double, m/s] : cap magnitude of final linear velocity (default: 2.0)
      alpha           [double, 0..1]: EMA smoothing for repulsion (1=no smoothing) (default: 1.0)
      ignore_z        [bool]        : if true, do APF in XY only (default: false; 3D by default)
      max_points      [int]         : random subsample limit from the cloud (default: 6000)
      attractive_topic[string]      : input topic (default: "~attractive_cmd")
      pointcloud_topic[string]      : input cloud (default: "~pointcloud")
      output_topic    [string]      : output twist (default: "~cmd")
    """

    def __init__(self):
        # Params
        self.r = rospy.get_param("~r", 2.0)
        self.eta_xy = rospy.get_param("~eta_xy", 0.8)
        self.eta_z = rospy.get_param("~eta_z", 0.8)
        self.eps = rospy.get_param("~eps", 0.10)
        self.max_rep_norm = rospy.get_param("~max_rep_norm", 1.0)
        self.max_out_norm = rospy.get_param("~max_out_norm", 2.0)
        self.alpha = rospy.get_param("~alpha", 1.0)
        self.ignore_z = rospy.get_param("~ignore_z", False)  # 3D by default
        self.max_points = int(rospy.get_param("~max_points", 6000))

        self.attractive_topic = rospy.get_param("~attractive_topic", "~attractive_cmd")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "~pointcloud")
        self.output_topic = rospy.get_param("~output_topic", "~cmd")

        # State
        self.last_attr = None  # TwistStamped
        self.rep_ema = np.zeros(3)  # for smoothing
        self.have_cloud = False

        # Pub/Sub
        self.pub = rospy.Publisher(self.output_topic, TwistStamped, queue_size=10)
        # NEW: Publisher for repulsive velocity
        self.rep_pub = rospy.Publisher("~repulsive_velocity", TwistStamped, queue_size=10)
        rospy.Subscriber(self.attractive_topic, TwistStamped, self.attr_cb, queue_size=10)
        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cloud_cb, queue_size=1)

        rospy.loginfo("[apf_3d_node] Ready. r=%.4f, eta_xy=%.4f, eta_z=%.4f, ignore_z=%s",
                      self.r, self.eta_xy, self.eta_z, self.ignore_z)

    # -------------------- Callbacks --------------------

    def attr_cb(self, msg: TwistStamped):
        self.last_attr = msg
        # Try to publish immediately using latest repulsion (from last cloud)
        self.publish_corrected()

    def cloud_cb(self, cloud_msg: PointCloud2):
        # Compute new repulsion from point cloud, update EMA
        rep = self.compute_repulsion(cloud_msg)
        if self.alpha <= 0.0:
            self.rep_ema = rep
        else:
            a = np.clip(self.alpha, 0.0, 1.0)
            self.rep_ema = a * rep + (1.0 - a) * self.rep_ema
        self.have_cloud = True

        # NEW: Publish repulsive velocity
        rep_twist = TwistStamped()
        rep_twist.header.stamp = rospy.Time.now()
        rep_twist.header.frame_id = cloud_msg.header.frame_id # Use cloud frame for repulsion
        rep_twist.twist.linear.x = float(self.rep_ema[0])
        rep_twist.twist.linear.y = float(self.rep_ema[1])
        rep_twist.twist.linear.z = float(self.rep_ema[2])
        self.rep_pub.publish(rep_twist)

        # Try to publish immediately using the latest attractive cmd
        self.publish_corrected()

    # -------------------- Core logic --------------------

    def compute_repulsion(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Vectorized APF repulsive velocity from PointCloud2."""
        # Gather points (x,y,z), skipping NaNs/inf
        pts_iter = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        # Convert to numpy
        # Efficiently sample up to max_points
        pts = []
        cnt = 0
        for p in pts_iter:
            x, y, z = p
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            pts.append((x, y, z))
            cnt += 1
            if cnt >= self.max_points:
                break

        if not pts:
            return np.zeros(3)

        P = np.asarray(pts, dtype=np.float64)  # shape (N,3)

        if self.ignore_z:
            # Project to XY plane for distance; set z to 0 in vector
            Pxy = P.copy()
            Pxy[:, 2] = 0.0
            d = np.linalg.norm(Pxy, axis=1)
            Pv = Pxy
        else:
            d = np.linalg.norm(P, axis=1)
            Pv = P

        # Mask points inside radius and beyond epsilon
        mask = (d < self.r) & (d > self.eps)
        if not np.any(mask):
            return np.zeros(3)

        d_sel = d[mask]
        Pv_sel = Pv[mask]  # vectors from robot to obstacle

        # Khatib repulsion with separate gains for XY and Z
        # Compute scalar weights per point, then sum -weights * p
        base_weights = (1.0 / d_sel - 1.0 / self.r) / (d_sel ** 3)  # shape (M,)
        
        # Apply different gains for XY and Z components
        rep_xy = -self.eta_xy * np.sum(Pv_sel[:, :2] * base_weights[:, None], axis=0)  # XY components
        rep_z = -self.eta_z * np.sum(Pv_sel[:, 2:3] * base_weights[:, None], axis=0)  # Z component
        rep_vec = np.array([rep_xy[0], rep_xy[1], rep_z[0]])

        # If we ignored z in distance, keep original z behavior:
        if self.ignore_z:
            rep_vec[2] = 0.0  # planar repulsion

        # Clip repulsion magnitude
        norm = np.linalg.norm(rep_vec)
        if norm > self.max_rep_norm > 0.0:
            rep_vec = rep_vec * (self.max_rep_norm / norm)

        return rep_vec

    def publish_corrected(self):
        if self.last_attr is None:
            return
        # Use latest repulsion (EMA) if we have received a cloud
        rep = self.rep_ema if self.have_cloud else np.zeros(3)

        # Attractive linear (xyz)
        v_attr = np.array([
            self.last_attr.twist.linear.x,
            self.last_attr.twist.linear.y,
            self.last_attr.twist.linear.z
        ], dtype=np.float64)

        # Final linear = attractive + repulsion
        v_out = v_attr + rep

        # Cap final output magnitude
        out_norm = np.linalg.norm(v_out)
        if out_norm > self.max_out_norm > 0.0:
            v_out = v_out * (self.max_out_norm / out_norm)

        # Build message
        out = TwistStamped()
        out.header.stamp = rospy.Time.now()
        # Keep the same frame as the attractive command to avoid frame confusion
        out.header.frame_id = self.last_attr.header.frame_id

        out.twist.linear.x = float(v_out[0])
        out.twist.linear.y = float(v_out[1])
        out.twist.linear.z = float(v_out[2])

        # Preserve yaw rate from attractive input, zero roll/pitch rates
        out.twist.angular.x = 0.0
        out.twist.angular.y = 0.0
        out.twist.angular.z = self.last_attr.twist.angular.z

        self.pub.publish(out)

def main():
    rospy.init_node("apf_3d_node")
    node = APF3DNode()
    rospy.spin()

if __name__ == "__main__":
    main()
