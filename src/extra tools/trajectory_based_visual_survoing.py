#!/usr/bin/env python3
import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.interpolate import CubicSpline

class VisualServoingTrajectoryNode:
    def __init__(self):
        rospy.init_node('visual_servoing_trajectory_node')

        self.velocity_topic = "/mavros/setpoint_velocity/cmd_vel_unstamped"
        self.visual_error_topic = "/visual_errors"
        self.yaw = 0

        # MAVROS-related attributes
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.local_pos_pub = rospy.Publisher(self.velocity_topic, Twist, queue_size=10)

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # Offboard and arming setup
        self.rate = rospy.Rate(20)
        self.pose = Twist()
        self.last_req = rospy.Time.now()

        # Trajectory generation setup
        self.current_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.target_position = np.array([0.0, 0.0, 1.5])   # Target at 1.5 meters away
        self.trajectory = None  # This will store the generated trajectory
        self.trajectory_index = 0
        self.trajectory_time = np.linspace(0, 10, 200)  # 10 seconds, 200 waypoints

        # Error subscriber
        rospy.Subscriber(self.visual_error_topic, Twist, self.visual_error_callback)

        # Ensure FCU connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        # Send a few setpoints before starting offboard mode
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

        self.pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)

    def state_cb(self, msg):
        self.current_state = msg

    def pose_callback(self, pose_msg):
        # Update current position from pose
        self.current_position = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z
        ])

        # Get the orientation quaternion and convert to yaw
        orientation_q = pose_msg.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def visual_error_callback(self, error_msg):
        error_x_meters = error_msg.linear.x
        error_y_meters = error_msg.linear.y
        distance_z = error_msg.linear.z
        yaw_error = error_msg.angular.z

        # If no trajectory, generate one
        if self.trajectory is None:
            self.generate_trajectory()

        # Follow the generated trajectory
        if self.trajectory is not None:
            self.follow_trajectory()

    def generate_trajectory(self):
        """Generate a smooth trajectory from the current position to the target position."""
        # Define start and end positions for x, y, and z
        start = self.current_position
        end = self.target_position

        # Cubic spline interpolation for smooth path generation
        spline_x = CubicSpline([0, 1], [start[0], end[0]])  # x-trajectory
        spline_y = CubicSpline([0, 1], [start[1], end[1]])  # y-trajectory
        spline_z = CubicSpline([0, 1], [start[2], end[2]])  # z-trajectory

        # Generate waypoints along the path
        x_traj = spline_x(np.linspace(0, 1, len(self.trajectory_time)))
        y_traj = spline_y(np.linspace(0, 1, len(self.trajectory_time)))
        z_traj = spline_z(np.linspace(0, 1, len(self.trajectory_time)))

        # Store the trajectory as waypoints
        self.trajectory = np.vstack((x_traj, y_traj, z_traj)).T

        rospy.loginfo("Trajectory generated")

    def follow_trajectory(self):
        """Follow the generated trajectory by adjusting velocity commands."""
        if self.trajectory_index < len(self.trajectory):
            target_position = self.trajectory[self.trajectory_index]

            # Compute position error with respect to the target position on the trajectory
            position_error = target_position - self.current_position

            # Proportional control for trajectory tracking
            k_x = 0.22
            k_y = 0.35
            k_z = 0.3
            k_yaw = 0.1

            # Velocity control based on position error
            body_x_vel = k_x * position_error[0]
            body_y_vel = k_y * position_error[1]
            body_z_vel = k_z * position_error[2]
            body_yaw_vel = k_yaw * 0.0  # No yaw error handling for now

            # Convert body frame velocity to global frame
            global_vel_x = body_x_vel * math.cos(self.yaw) - body_y_vel * math.sin(self.yaw)
            global_vel_y = body_x_vel * math.sin(self.yaw) + body_y_vel * math.cos(self.yaw)
            global_vel_z = body_z_vel

            # Publish velocity commands
            self.pose.linear.x = global_vel_x
            self.pose.linear.y = global_vel_y
            self.pose.linear.z = global_vel_z
            self.pose.angular.z = body_yaw_vel
            self.local_pos_pub.publish(self.pose)

            # Increment the trajectory index to move to the next waypoint
            self.trajectory_index += 1
        else:
            rospy.loginfo("Trajectory completed")

    def run(self):
        while not rospy.is_shutdown():
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req > rospy.Duration(5.0)):
                if self.set_mode_client.call(self.offb_set_mode).mode_sent:
                    rospy.loginfo("Offboard enabled")
                self.last_req = rospy.Time.now()
            elif not self.current_state.armed and (rospy.Time.now() - self.last_req > rospy.Duration(5.0)):
                if self.arming_client.call(self.arm_cmd).success:
                    rospy.loginfo("Vehicle armed")
                self.last_req = rospy.Time.now()

            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = VisualServoingTrajectoryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
