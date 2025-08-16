#!/usr/bin/env python
import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
import tf

class IntegratedPlannerNode(object):
    def __init__(self):
        rospy.init_node('integrated_planner_node', anonymous=True)

        # Subscribers for current pose and goal.
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # Publishers for visualization and for sending the next waypoint.
        self.global_path_pub = rospy.Publisher('global_path', Path, queue_size=1)
        self.simplified_path_pub = rospy.Publisher('simplified_path', Path, queue_size=1)
        self.next_waypoint_pub = rospy.Publisher('next_waypoint', PoseStamped, queue_size=1)

        self.current_pose = None
        self.goal_pose = None
        self.global_path = None
        self.key_waypoints = []  # List of key waypoints (excluding the current pose)
        self.current_waypoint_index = 0

        # Parameter: distance threshold to consider a waypoint reached (meters)
        self.waypoint_threshold = rospy.get_param("~waypoint_threshold", 0.5)

        rospy.loginfo("Integrated planner node initialized.")

    def pose_callback(self, msg):
        self.current_pose = msg
        self.check_waypoint_reached()

    def goal_callback(self, msg):
        rospy.loginfo("Received new goal: (%.2f, %.2f, %.2f)",
                      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.goal_pose = msg
        self.compute_global_path()

    def compute_global_path(self):
        if self.current_pose is None or self.goal_pose is None:
            rospy.logwarn("Current pose or goal not available. Cannot compute global path.")
            return

        start = self.current_pose.pose.position
        goal = self.goal_pose.pose.position

        num_points = 20  # Adjust as needed.
        xs = np.linspace(start.x, goal.x, num_points)
        ys = np.linspace(start.y, goal.y, num_points)
        zs = np.linspace(start.z, goal.z, num_points)

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.current_pose.header.frame_id

        last_yaw = 0.0
        for i in range(num_points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = xs[i]
            pose.pose.position.y = ys[i]
            pose.pose.position.z = zs[i]
            # Compute yaw based on difference from previous point
            if i > 0:
                dx = xs[i] - xs[i-1]
                dy = ys[i] - ys[i-1]
                yaw = math.atan2(dy, dx)
            else:
                yaw = last_yaw
            pose.pose.orientation = self.yaw_to_quaternion(yaw)
            last_yaw = yaw
            path_msg.poses.append(pose)

        # Check ordering: if the first point is farther than the last, reverse the path.
        dist_first = self.euclidean_distance(self.current_pose.pose.position, path_msg.poses[0].pose.position)
        dist_last = self.euclidean_distance(self.current_pose.pose.position, path_msg.poses[-1].pose.position)
        if dist_first > dist_last:
            rospy.loginfo("Global path appears reversed. Reversing it.")
            path_msg.poses.reverse()

        self.global_path = path_msg
        self.global_path_pub.publish(path_msg)
        rospy.loginfo("Global path computed with %d points", len(path_msg.poses))
        self.extract_key_waypoints()

    def extract_key_waypoints(self):
        """
        Simplify the global path to extract key waypoints.
        In this revision, we skip the first point (current pose) to avoid immediate waypoint reach.
        """
        if self.global_path is None or not self.global_path.poses:
            rospy.logwarn("No global path to simplify.")
            return

        sample_rate = 5  # For example, take every 5th point from the global path (starting from index 1)
        key_waypoints = []
        # Start from index 1 to skip the current position.
        for i in range(1, len(self.global_path.poses), sample_rate):
            key_waypoints.append(self.global_path.poses[i])
        # Ensure the final goal is included:
        if self.global_path.poses[-1] != key_waypoints[-1]:
            key_waypoints.append(self.global_path.poses[-1])
        
        self.key_waypoints = key_waypoints
        self.current_waypoint_index = 0

        simplified_path = Path()
        simplified_path.header = self.global_path.header
        simplified_path.poses = self.key_waypoints
        self.simplified_path_pub.publish(simplified_path)
        rospy.loginfo("Simplified path has %d key waypoints.", len(self.key_waypoints))

        self.publish_next_waypoint()

    def publish_next_waypoint(self):
        if self.current_waypoint_index < len(self.key_waypoints):
            next_wp = self.key_waypoints[self.current_waypoint_index]
            self.next_waypoint_pub.publish(next_wp)
            rospy.loginfo("Publishing waypoint %d: (%.2f, %.2f, %.2f)",
                          self.current_waypoint_index,
                          next_wp.pose.position.x, next_wp.pose.position.y, next_wp.pose.position.z)
        else:
            rospy.loginfo("No more waypoints to publish (final goal reached).")

    def check_waypoint_reached(self):
        if self.current_pose is None or not self.key_waypoints:
            return

        target_wp = self.key_waypoints[self.current_waypoint_index]
        dx = self.current_pose.pose.position.x - target_wp.pose.position.x
        dy = self.current_pose.pose.position.y - target_wp.pose.position.y
        dz = self.current_pose.pose.position.z - target_wp.pose.position.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        rospy.loginfo("Distance to waypoint %d: %.2f m", self.current_waypoint_index, dist)
        if dist < self.waypoint_threshold:
            rospy.loginfo("Waypoint %d reached (distance %.2f m).", self.current_waypoint_index, dist)
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.key_waypoints):
                self.publish_next_waypoint()
            else:
                rospy.loginfo("Final waypoint reached. Mission complete.")

    def yaw_to_quaternion(self, yaw):
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def euclidean_distance(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

if __name__ == '__main__':
    try:
        node = IntegratedPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
