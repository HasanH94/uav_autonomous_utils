#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import threading
import tf.transformations
from std_msgs.msg import Bool
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State

class HybridYawPIDController:
    def __init__(self):
        rospy.init_node('sliding_mode_vel_controller')
        self.max_speed = rospy.get_param('~max_speed', 2.5)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 2.0) # Max yaw rate in rad/s
        self.yaw_switch_distance = rospy.get_param('~yaw_switch_distance', 1.0) # Distance to goal to switch yaw control

        # PID Gains for Yaw
        self.kp_yaw = rospy.get_param('~kp_yaw', 0.8)
        self.ki_yaw = rospy.get_param('~ki_yaw', 0.0)
        self.kd_yaw = rospy.get_param('~kd_yaw', 0.1)

        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1) # Position tolerance for goal reached
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.1) # Yaw tolerance for goal reached
        self.is_gps_navigation_mode = rospy.get_param('~is_gps_navigation_mode', False) # True if used for GPS navigation

        self.goal = None
        self.goal_orientation = None
        self.current_pose = None
        self.current_vel = np.zeros(3)
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_yaw_rate = 0.0
        self.apf_linear_velocity = np.zeros(3)

        # PID state variables for Yaw
        self.integral_yaw = 0.0
        self.previous_error_yaw = 0.0
        self.last_time = rospy.Time.now()

        self.mavros_current_state = State()
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb)
        rospy.Subscriber('/apf_vel', TwistStamped, self.apf_vel_cb)
        self.pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/drone_events/reached_desired_pose', Bool, queue_size=1)
        self.gps_goal_reached_pub = rospy.Publisher('/drone_events/reached_goal_point', Bool, queue_size=1)

        self.lock = threading.Lock()
        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.shutdown_hook)
        self.set_offboard_mode()
        self.control_loop()

    def mavros_state_callback(self, msg):
        self.mavros_current_state = msg

    def set_offboard_mode(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        last_req = rospy.Time.now()

        rospy.loginfo("Attempting to set OFFBOARD mode...")
        # Send a few setpoints before attempting to switch mode
        # This is crucial for PX4 to accept OFFBOARD mode
        for i in range(100):
            if rospy.is_shutdown():
                return
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = "odom"
            self.pub.publish(twist)
            self.rate.sleep()

        while not rospy.is_shutdown() and self.mavros_current_state.mode != "OFFBOARD":
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled.")
                else:
                    rospy.logwarn_throttle(5, "Failed to set OFFBOARD mode.")
                last_req = rospy.Time.now()

            # Publish setpoints even while waiting for OFFBOARD mode to be set
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = "odom"
            self.pub.publish(twist)
            self.rate.sleep()
        rospy.loginfo("Drone is in OFFBOARD mode.")

    

    def control_loop(self):
        while not rospy.is_shutdown():
            with self.lock:
                if self.goal is None or self.current_pose is None:
                    self.rate.sleep()
                    continue

                current_time = rospy.Time.now()
                dt = (current_time - self.last_time).to_sec()
                self.last_time = current_time

                if dt <= 0:
                    self.rate.sleep()
                    continue

                # Position Errors
                pos_error = self.goal - self.current_pose
                distance_to_goal = np.linalg.norm(pos_error)

                # Yaw Error (hybrid control)
                horizontal_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
                desired_yaw = self.current_yaw # Default to current yaw if no goal or close enough

                if horizontal_dist > self.yaw_switch_distance:
                    # Far from goal: point towards the goal position
                    delta_x = self.goal[0] - self.current_pose[0]
                    delta_y = self.goal[1] - self.current_pose[1]
                    if abs(delta_x) > 0.01 or abs(delta_y) > 0.01: # Avoid division by zero or erratic behavior near goal
                        desired_yaw = np.arctan2(delta_y, delta_x)
                elif self.goal_orientation is not None:
                    # Close to goal: align with the published goal orientation
                    goal_orientation_list = [self.goal_orientation.x, self.goal_orientation.y, self.goal_orientation.z, self.goal_orientation.w]
                    (roll_g, pitch_g, desired_yaw) = tf.transformations.euler_from_quaternion(goal_orientation_list)
                
                e_yaw = desired_yaw - self.current_yaw
                e_yaw = np.arctan2(np.sin(e_yaw), np.cos(e_yaw)) # Normalize yaw error
                # rospy.loginfo(f"Yaw Error: {e_yaw:.4f}")

                # PID Calculation for Yaw only
                self.integral_yaw += e_yaw * dt
                self.integral_yaw = np.clip(self.integral_yaw, -1.0, 1.0) # Anti-windup
                derivative_yaw = (e_yaw - self.previous_error_yaw) / dt
                output_yaw = (self.kp_yaw * e_yaw) + (self.ki_yaw * self.integral_yaw) + (self.kd_yaw * derivative_yaw)
                self.previous_error_yaw = e_yaw

                yaw_rate = np.clip(output_yaw, -self.max_yaw_rate, self.max_yaw_rate)

                # Calculate linear velocity magnitude
                linear_velocity_magnitude = np.linalg.norm(self.apf_linear_velocity)

                # Limit linear velocity if it exceeds max_speed
                if linear_velocity_magnitude > self.max_speed:
                    self.apf_linear_velocity = (self.apf_linear_velocity / linear_velocity_magnitude) * self.max_speed

                # Combine APF linear velocity with PID yaw rate and publish
                twist = TwistStamped()
                twist.header.stamp = current_time
                twist.header.frame_id = "odom"
                twist.twist.linear.x = self.apf_linear_velocity[0]
                twist.twist.linear.y = self.apf_linear_velocity[1]
                twist.twist.linear.z = self.apf_linear_velocity[2]
                twist.twist.angular.z = yaw_rate
                self.pub.publish(twist)

                # rospy.loginfo(f"Total Position Error: {distance_to_goal:.4f}")

                # Check if goal is reached and publish status
                if distance_to_goal < self.position_tolerance and abs(e_yaw) < self.yaw_tolerance:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(True)
                        rospy.loginfo("GPS Goal Reached: Position and Yaw within tolerance.")
                    else:
                        self.goal_reached_pub.publish(True)
                        rospy.loginfo("Visual Servoing Goal Reached: Position and Yaw within tolerance.")
                else:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(False)
                    else:
                        self.goal_reached_pub.publish(False)

            self.rate.sleep()

    def goal_cb(self, msg):
        with self.lock:
            self.goal = np.array([msg.pose.position.x,
                                  msg.pose.position.y,
                                  msg.pose.position.z])
            self.goal_orientation = msg.pose.orientation

    def apf_vel_cb(self, msg):
        with self.lock:
            self.apf_linear_velocity[0] = msg.twist.linear.x
            self.apf_linear_velocity[1] = msg.twist.linear.y
            self.apf_linear_velocity[2] = msg.twist.linear.z

    def odom_cb(self, msg):
        with self.lock:
            self.current_pose = np.array([msg.pose.pose.position.x,
                                          msg.pose.pose.position.y,
                                          msg.pose.pose.position.z])
            self.current_vel = np.array([msg.twist.twist.linear.x,
                                         msg.twist.twist.linear.y,
                                         msg.twist.twist.linear.z])
            # Extract yaw from quaternion
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
            self.current_roll = roll
            self.current_pitch = pitch
            self.current_yaw = yaw
            self.current_yaw_rate = msg.twist.twist.angular.z

    def shutdown_hook(self):
        rospy.loginfo("HybridYawPIDController is shutting down. Setting mode to AUTO.LOITER.")
        set_mode_srv = SetModeRequest()
        set_mode_srv.custom_mode = 'AUTO.LOITER'
        try:
            if self.set_mode_client.call(set_mode_srv).mode_sent:
                rospy.loginfo("Successfully set to AUTO.LOITER mode.")
            else:
                rospy.logwarn("Failed to set AUTO.LOITER mode on shutdown.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed on shutdown: {e}")

if __name__ == '__main__':
    controller = HybridYawPIDController()
