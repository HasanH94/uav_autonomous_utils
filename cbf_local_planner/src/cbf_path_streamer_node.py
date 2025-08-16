#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import rospy
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Transform, Vector3, Quaternion
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


class PathStreamer:
    def __init__(self):
        rospy.init_node('cbf_path_streamer_node')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.enable_preview = rospy.get_param('~enable_preview', False)
        self.preview_horizon_s = rospy.get_param('~preview_horizon_s', 1.2)
        self.preview_points = rospy.get_param('~preview_points', 4)
        self.pub_rate_hz = rospy.get_param('~pub_rate_hz', 15.0)

        self.pose = None
        self.yaw = 0.0
        self.last_cmd = Twist()
        self.goal_reached = False

        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber('/cbf/last_cmd', TwistStamped, self.cmd_cb, queue_size=1)
        rospy.Subscriber('/cbf/goal_reached', Bool, self.goal_reached_cb, queue_size=1)
        self.pub_path = rospy.Publisher('/cbf/preview_path', Path, queue_size=1)
        self.pub_traj = rospy.Publisher('/mavros/setpoint_trajectory/input', MultiDOFJointTrajectory, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0/max(1e-3,self.pub_rate_hz)), self.timer_cb)

    def odom_cb(self, msg: Odometry):
        self.pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = msg.pose.pose.orientation
        (_,_,self.yaw) = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])

    def cmd_cb(self, msg: TwistStamped):
        self.last_cmd = msg.twist
    
    def goal_reached_cb(self, msg: Bool):
        self.goal_reached = msg.data

    def timer_cb(self, _):
        if self.pose is None:
            return
        # Build a path by integrating last_cmd (for viz + simple streaming)
        path = Path(); path.header.stamp = rospy.Time.now(); path.header.frame_id = self.odom_frame
        dt = self.preview_horizon_s / float(max(1, self.preview_points-1))
        pos = self.pose[:]; yaw = self.yaw
        for i in range(self.preview_points):
            if i>0:
                pos[0] += self.last_cmd.linear.x * dt
                pos[1] += self.last_cmd.linear.y * dt
                pos[2] += self.last_cmd.linear.z * dt
                yaw    += self.last_cmd.angular.z * dt
            ps = PoseStamped(); ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos[0], pos[1], pos[2]
            qy = tft.quaternion_from_euler(0,0,yaw)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qy
            path.poses.append(ps)
        self.pub_path.publish(path)

        if not self.enable_preview:
            return
        
        # Stream trajectory - if goal reached, send stationary point at current position
        if self.goal_reached:
            # Send stationary trajectory at current position to maintain >2Hz stream
            traj = MultiDOFJointTrajectory()
            traj.header.stamp = rospy.Time.now()
            traj.header.frame_id = self.odom_frame
            traj.joint_names = ['base']
            
            tfm = Transform()
            tfm.translation = Vector3(self.pose[0], self.pose[1], self.pose[2])
            qy = tft.quaternion_from_euler(0, 0, self.yaw)
            tfm.rotation = Quaternion(qy[0], qy[1], qy[2], qy[3])
            
            pt = MultiDOFJointTrajectoryPoint(
                transforms=[tfm],
                velocities=[Twist()],  # zero velocity
                accelerations=[Twist()],
                time_from_start=rospy.Duration.from_sec(0.0)
            )
            traj.points.append(pt)
            self.pub_traj.publish(traj)
            return
        
        # Stream a simple MultiDOFJointTrajectory along the integrated points
        traj = MultiDOFJointTrajectory()
        traj.header = path.header
        traj.joint_names = ['base']
        tfs = []; vels = []; accs = []
        for i, ps in enumerate(path.poses):
            tfm = Transform()
            tfm.translation = Vector3(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z)
            tfm.rotation = Quaternion(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w)
            tfs.append(tfm)
            # Provide velocities (optional but nice)
            v = Twist(); v.linear = self.last_cmd.linear; v.angular = self.last_cmd.angular
            vels.append(v)
            accs.append(Twist())
            pt = MultiDOFJointTrajectoryPoint(transforms=[tfm], velocities=[v], accelerations=[Twist()],
                                              time_from_start=rospy.Duration.from_sec(i*dt))
            traj.points.append(pt)
        self.pub_traj.publish(traj)


if __name__ == '__main__':
    PathStreamer()
    rospy.spin()