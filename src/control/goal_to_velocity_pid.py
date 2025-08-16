#!/usr/bin/env python
import rospy
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State
from std_msgs.msg import Bool # For publishing events
from tf.transformations import euler_from_quaternion

class GoalToVelocityPID:
    def __init__(self):
        rospy.init_node('goal_to_velocity_pid')

        # State Machine
        self.STATE_NAVIGATING = "NAVIGATING"
        self.STATE_ROTATING_TO_GOAL_YAW = "ROTATING_TO_GOAL_YAW"
        self.STATE_HOLDING_POSITION = "HOLDING_POSITION"
        self.current_state = self.STATE_NAVIGATING

        # Parameters
        self.target_x = rospy.get_param('~target_x', 5.0)
        self.target_y = rospy.get_param('~target_y', 0.0)
        self.target_z = rospy.get_param('~target_z', 1.0)
        self.max_speed = rospy.get_param('~max_speed', 2.0)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 0.5)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.1)
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.05) # Radians, ~3 degrees
        self.hover_mode = rospy.get_param('~hover_mode', "AUTO.LOITER")
        self.enable_yaw_alignment = rospy.get_param('~enable_yaw_alignment', False)
        self.min_goal_change = rospy.get_param('~min_goal_change', 0.05) # Minimum change in position or yaw to accept a new goal

        # PID Gains
        self.kp = np.array(rospy.get_param('~kp', [0.5, 0.5, 0.5, 0.8]))
        self.ki = np.array(rospy.get_param('~ki', [0.01, 0.01, 0.01, 0.0]))
        self.kd = np.array(rospy.get_param('~kd', [0.1, 0.1, 0.1, 0.1]))

        self.target_pos = np.array([self.target_x, self.target_y, self.target_z])
        self.target_yaw = 0.0

        # PID state variables
        self.integral = np.zeros(4)
        self.previous_error = np.zeros(4)
        self.last_time = rospy.Time.now()

        self.current_pos = np.zeros(3)
        self.current_yaw = 0.0
        self.mavros_current_state = State()

        # ROS Services, Subscribers and Publishers
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        self.vel_pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=10)
        self.goal_reached_event_pub = rospy.Publisher('/drone_events/reached_goal_point', Bool, queue_size=1)

        self.rate = rospy.Rate(20.0)
        rospy.on_shutdown(self.shutdown_cb)

        rospy.loginfo("GoalToVelocityPID controller initialized with rotation step.")

    def odom_callback(self, msg):
        self.current_pos[0] = msg.pose.pose.position.x
        self.current_pos[1] = msg.pose.pose.position.y
        self.current_pos[2] = msg.pose.pose.position.z
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def mavros_state_callback(self, msg):
        self.mavros_current_state = msg

    def goal_callback(self, msg):
        new_target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        q = msg.pose.orientation
        _, _, new_target_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if np.linalg.norm(new_target_pos - self.target_pos) > self.min_goal_change or abs(new_target_yaw - self.target_yaw) > self.yaw_tolerance:
            self.target_pos = new_target_pos
            self.target_yaw = new_target_yaw
            rospy.loginfo(f"New goal received: Pos={self.target_pos}, Yaw={self.target_yaw:.2f}")
            if self.current_state != self.STATE_NAVIGATING:
                self.switch_state(self.STATE_NAVIGATING)
                self.set_offboard_mode()
        else:
            rospy.loginfo_throttle(5, "Ignoring new goal, too close to current target.")

    def switch_state(self, new_state):
        if self.current_state != new_state:
            rospy.loginfo(f"Switching from {self.current_state} to {new_state}")
            self.current_state = new_state

    def set_mode(self, mode):
        try:
            set_mode_req = SetModeRequest()
            set_mode_req.custom_mode = mode
            resp = self.set_mode_client.call(set_mode_req)
            if resp.mode_sent:
                rospy.loginfo(f"Mode change to {mode} successful.")
                return True
            else:
                rospy.logwarn(f"Failed to send mode change request for {mode}.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set_mode failed: {e}")
            return False

    def wait_for_connection(self):
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.mavros_current_state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected.")

    def set_offboard_mode(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        last_req = rospy.Time.now()
        
        rospy.loginfo("Publishing setpoints and attempting to set OFFBOARD mode...")
        zero_vel = TwistStamped()
        zero_vel.header.frame_id = "map"

        while not rospy.is_shutdown() and self.mavros_current_state.mode != "OFFBOARD":
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled.")
                else:
                    rospy.logwarn_throttle(5, "Failed to set OFFBOARD mode.")
                last_req = rospy.Time.now()

            zero_vel.header.stamp = rospy.Time.now()
            self.vel_pub.publish(zero_vel)
            self.rate.sleep()
        rospy.loginfo("Drone is in OFFBOARD mode.")

    def shutdown_cb(self):
        rospy.loginfo("Node is shutting down. Switching to hover mode.")
        self.set_mode(self.hover_mode)
        
        t_end = rospy.Time.now() + rospy.Duration(1.0)
        zero_vel = TwistStamped()
        zero_vel.header.frame_id = "map"
        while rospy.Time.now() < t_end and not rospy.is_shutdown():
            zero_vel.header.stamp = rospy.Time.now()
            self.vel_pub.publish(zero_vel)
            self.rate.sleep()

    def run(self):
        self.wait_for_connection()
        self.set_offboard_mode()

        while not rospy.is_shutdown():
            if self.mavros_current_state.mode != "OFFBOARD" and self.current_state != self.STATE_HOLDING_POSITION:
                rospy.logerr("Aborting control: Not in OFFBOARD mode. Switching to HOLD.")
                self.set_mode(self.hover_mode)
                self.switch_state(self.STATE_HOLDING_POSITION)

            if self.current_state == self.STATE_NAVIGATING:
                self.navigate_loop()
            elif self.current_state == self.STATE_ROTATING_TO_GOAL_YAW:
                self.rotate_loop()
            elif self.current_state == self.STATE_HOLDING_POSITION:
                rospy.loginfo_throttle(5, "Holding position at goal. Waiting for new goal.")
                
            self.rate.sleep()

    def navigate_loop(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt <= 0: return

        pos_error = self.target_pos - self.current_pos
        distance_to_goal = np.linalg.norm(pos_error)
        rospy.loginfo_throttle(1.0, f"NAVIGATING: Dist: {distance_to_goal:.2f}m")

        if distance_to_goal <= self.goal_tolerance:
            # Reset PID integrals as we are done with navigation
            self.integral = np.zeros(4)
            self.previous_error = np.zeros(4)

            if self.enable_yaw_alignment:
                rospy.loginfo("Position reached. Switching to rotation.")
                self.switch_state(self.STATE_ROTATING_TO_GOAL_YAW)
            else:
                rospy.loginfo("Goal reached. Yaw alignment disabled. Switching to hover mode.")
                self.set_mode(self.hover_mode)
                self.goal_reached_event_pub.publish(True)
                self.switch_state(self.STATE_HOLDING_POSITION)
            return

        # Calculate yaw error to face the goal point
        desired_yaw = math.atan2(pos_error[1], pos_error[0])
        yaw_error = desired_yaw - self.current_yaw
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi

        error = np.array([pos_error[0], pos_error[1], pos_error[2], yaw_error])
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        derivative = (error - self.previous_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error

        linear_output = output[:3]
        speed = np.linalg.norm(linear_output)
        if speed > self.max_speed:
            linear_output = (linear_output / speed) * self.max_speed
        yaw_rate = np.clip(output[3], -self.max_yaw_rate, self.max_yaw_rate)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = current_time
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = linear_output[0]
        vel_msg.twist.linear.y = linear_output[1]
        vel_msg.twist.linear.z = linear_output[2]
        vel_msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(vel_msg)

    def rotate_loop(self):
        yaw_error = self.target_yaw - self.current_yaw
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi

        rospy.loginfo_throttle(0.5, f"ROTATING: Yaw Error: {yaw_error:.2f} rad")

        if abs(yaw_error) <= self.yaw_tolerance:
            rospy.loginfo("Yaw aligned. Goal reached. Switching to hover mode.")
            # Publish one last zero velocity command to stop rotation
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.header.frame_id = "map"
            self.vel_pub.publish(vel_msg) # Zero linear and angular
            
            self.set_mode(self.hover_mode)
            self.goal_reached_event_pub.publish(True)
            self.switch_state(self.STATE_HOLDING_POSITION)
            return

        # Simple P-controller for yaw rate
        yaw_rate = self.kp[3] * yaw_error
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = 0
        vel_msg.twist.linear.y = 0
        vel_msg.twist.linear.z = 0
        vel_msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(vel_msg)

if __name__ == '__main__':
    try:
        controller = GoalToVelocityPID()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
        pass