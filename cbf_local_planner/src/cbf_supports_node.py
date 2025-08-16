#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import threading
from typing import List, Optional

import numpy as np
import rospy
import tf
import tf.transformations as tft

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from uav_autonomous_utils.msg import SupportPoint, SupportArray


def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class CloudRingBuffer:
    def __init__(self, max_len=8, crop_radius=5.0, stride=4):
        self.max_len = max_len
        self.crop_r2 = crop_radius * crop_radius
        self.stride = max(1, int(stride))
        self.lock = threading.Lock()
        self.buff = []   # list of (N_i x 3) np.float32 in odom
        self.stamps = [] # float (sec)

    def add_cloud(self, cloud_msg: PointCloud2, tf_listener: tf.TransformListener, odom_frame: str) -> Optional[float]:
        if cloud_msg is None:
            return None
        # Decimate + crop in sensor frame
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
            self.stamps.append(cloud_msg.header.stamp.to_sec())
            if len(self.buff) > self.max_len:
                self.buff.pop(0)
                self.stamps.pop(0)
        return self.stamps[-1]

    def nearest_supports(self, q_odom: np.ndarray, k: int, region_radius: float, min_neighbors: int):
        with self.lock:
            if not self.buff:
                return []
            cloud = self.buff[-1]
            stamps = list(self.stamps)

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
            g = (q_odom - p) / max(d, 1e-6)  # outward normal in odom

            # Neighborhood in current cloud
            nbr_mask = np.linalg.norm(cloud - p.reshape(1,3), axis=1) <= region_radius
            nbrs = cloud[nbr_mask]
            if nbrs.shape[0] < min_neighbors:
                continue

            # PCA surface variation
            mu = np.mean(nbrs, axis=0)
            X = nbrs - mu
            C = (X.T @ X) / max(1, X.shape[0]-1)
            evals, _ = np.linalg.eigh(C)
            evals = np.clip(evals, 0.0, None)
            surf_var = float(evals[0] / max(1e-9, np.sum(evals)))

            # Temporal persistence + variance along g over past clouds
            hits = 0; total = 0; proj_vals = []
            with self.lock:
                for pc_arr in self.buff[:-1]:
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
                var_along_g = 0.05

            # Confidence kappa
            tau = rospy.get_param('~kappa_tau', 0.5)
            a = rospy.get_param('~kappa_slope', 6.0)
            sigma0 = rospy.get_param('~kappa_sigma0', 0.09)
            pers_term = 1.0 / (1.0 + math.exp(-a*(persistence - tau)))
            var_term  = math.exp(-var_along_g / max(1e-6, sigma0))
            kappa = pers_term * var_term * (1.0 - clip(surf_var*3.0, 0.0, 1.0))
            kappa = float(clip(kappa, 0.0, 1.0))

            sp = SupportPoint()
            sp.p.x, sp.p.y, sp.p.z = float(p[0]), float(p[1]), float(p[2])
            sp.g.x, sp.g.y, sp.g.z = float(g[0]), float(g[1]), float(g[2])
            sp.d = d
            sp.kappa = kappa
            supports.append(sp)
            chosen += 1
            if chosen >= k:
                break
        return supports


class SupportsNode:
    def __init__(self):
        rospy.init_node('cbf_supports_node')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.support_k = rospy.get_param('~support_k', 1)
        self.region_radius = rospy.get_param('~region_radius', 0.20)
        self.min_neighbors = rospy.get_param('~min_neighbors', 12)
        self.crop_radius = rospy.get_param('~crop_radius', 5.0)
        self.stride = rospy.get_param('~stride', 4)
        self.buf_len = rospy.get_param('~buffer_len', 8)

        self.tf_listener = tf.TransformListener()
        self.ring = CloudRingBuffer(self.buf_len, self.crop_radius, self.stride)

        rospy.Subscriber('/camera/depth/points', PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb, queue_size=1)
        self.pub = rospy.Publisher('/cbf/supports', SupportArray, queue_size=1)

        self.q = None
        self.last_depth_stamp = None
        self.rate = rospy.Rate(30)

    def cloud_cb(self, msg: PointCloud2):
        ts = self.ring.add_cloud(msg, self.tf_listener, self.odom_frame)
        if ts is not None:
            self.last_depth_stamp = ts

    def odom_cb(self, msg: Odometry):
        self.q = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float64)

    def spin(self):
        while not rospy.is_shutdown():
            if self.q is not None and self.last_depth_stamp is not None:
                s_list = self.ring.nearest_supports(self.q, self.support_k, self.region_radius, self.min_neighbors)
                arr = SupportArray()
                arr.header.stamp = rospy.Time.from_sec(self.last_depth_stamp)
                arr.header.frame_id = self.odom_frame
                arr.supports = s_list
                self.pub.publish(arr)
            self.rate.sleep()


if __name__ == '__main__':
    node = SupportsNode()
    node.spin()