#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from tf.transformations import euler_from_quaternion

class VisualServoingPIDNode:
    def __init__(self):
        rospy.init_node('visual_servoing_pid_node')

        # Topics
        self.velocity_topic = "/mavros/setpoint_velocity/cmd_vel_unstamped"
        self.visual_error_topic = "/visual_errors"
        self.pose_topic = "/mavros/local_position/pose"

        # Internal yaw tracking
        self.yaw = 0.0

        # MAVROS State
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        # Publisher for velocity commands
        self.local_pos_pub = rospy.Publisher(self.velocity_topic, Twist, queue_size=10)

        # Subscribe to local_position (for orientation -> yaw)
        self.pose_subscriber = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)

        # Wait for services
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # Initialize
        self.rate = rospy.Rate(20)  # 20 Hz
        self.pose_msg = Twist()     # The velocity setpoint we publish

        # Prepare OFFBOARD & arm commands
        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

        self.last_req = rospy.Time.now()

        # PID control variables
        self.integral_error_x = 0.0
        self.integral_error_y = 0.0
        self.integral_error_z = 0.0
        self.integral_error_yaw = 0.0

        # Error history for derivative control
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        self.prev_error_yaw = 0.0

        self.dt = 0.05  # Time step for derivative control, ~ 20 Hz

        # Subscribe to visual errors
        rospy.Subscriber(self.visual_error_topic, Twist, self.visual_error_callback)

        # Wait for FCU connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("[visual_servoing_pid_node] FCU connected.")

        # Send a few zero setpoints before switching to OFFBOARD
        rospy.loginfo("[visual_servoing_pid_node] Publishing initial zero setpoints...")
        zero_twist = Twist()
        for _ in range(100):
            if rospy.is_shutdown():
                return
            self.local_pos_pub.publish(zero_twist)
            self.rate.sleep()
        rospy.loginfo("[visual_servoing_pid_node] Done publishing initial zero setpoints.")

    def state_cb(self, msg):
        self.current_state = msg

    def pose_callback(self, pose_msg):
        """Extract yaw from the local_position pose (quaternion -> euler)."""
        orientation_q = pose_msg.pose.orientation
        _, _, self.yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

    def visual_error_callback(self, error_msg):
        """
        Called whenever /visual_errors is published.
        We'll compute velocities in body frame, convert to global,
        and publish to /mavros/setpoint_velocity/cmd_vel_unstamped.
        """
        # Extract errors
        error_x_meters = error_msg.linear.x   # e.g. x-offset
        error_y_meters = error_msg.linear.y   # y-offset
        distance_z      = error_msg.linear.z   # z-offset or something
        yaw_error       = error_msg.angular.z

        # For logging / debugging
        distance = math.sqrt(error_x_meters**2 + error_y_meters**2 + distance_z**2)

        # Gains
        k_x   = 0.22
        k_y   = 0.35
        k_z   = 0.3
        k_yaw = 0.1

        kd_x   = 0.0
        kd_y   = 0.005
        kd_z   = 0.0
        kd_yaw = 0.03

        integral_gain_x   = 0.0
        integral_gain_y   = 0.0
        integral_gain_z   = 0.0
        integral_gain_yaw = 0.0

        # Update integrals
        self.integral_error_x   += error_x_meters * self.dt
        self.integral_error_y   += error_y_meters * self.dt
        self.integral_error_z   += distance_z * self.dt
        self.integral_error_yaw += yaw_error * self.dt

        # Derivatives
        derivative_x   = (error_x_meters - self.prev_error_x) / self.dt
        derivative_y   = (error_y_meters - self.prev_error_y) / self.dt
        derivative_z   = (distance_z      - self.prev_error_z) / self.dt
        derivative_yaw = (yaw_error       - self.prev_error_yaw) / self.dt

        # Construct body-frame velocities:
        # Possibly your logic is: X velocity depends on distance_z, etc.
        body_x_vel = k_x * distance_z + kd_x * derivative_z + integral_gain_x * self.integral_error_z
        body_y_vel = k_y * error_x_meters + kd_y * derivative_x + integral_gain_y * self.integral_error_x
        body_z_vel = k_z * error_y_meters + kd_z * derivative_y + integral_gain_z * self.integral_error_y
        body_yaw_vel = k_yaw * yaw_error + kd_yaw * derivative_yaw + integral_gain_yaw * self.integral_error_yaw

        # Convert body-frame velocities to global-frame velocities using self.yaw
        global_vel_x = body_x_vel * math.cos(self.yaw) - body_y_vel * math.sin(self.yaw)
        global_vel_y = body_x_vel * math.sin(self.yaw) + body_y_vel * math.cos(self.yaw)
        global_vel_z = body_z_vel
        global_vel_yaw = body_yaw_vel

        # Example "stop" condition if close enough:
        # If distance < 1.5 and the forward velocity is positive, stop
        if (distance < 1.5 and body_x_vel > 0):
            self.pose_msg.linear.x = 0.0
            self.pose_msg.linear.y = 0.0
            self.pose_msg.linear.z = 0.0
            self.pose_msg.angular.z = 0.0
        else:
            self.pose_msg.linear.x = global_vel_x
            self.pose_msg.linear.y = global_vel_y
            self.pose_msg.linear.z = global_vel_z
            self.pose_msg.angular.z = global_vel_yaw

        # Publish
        self.local_pos_pub.publish(self.pose_msg)

        # Update previous error for derivative
        self.prev_error_x = error_x_meters
        self.prev_error_y = error_y_meters
        self.prev_error_z = distance_z
        self.prev_error_yaw = yaw_error

    def run(self):
        """
        Main loop: tries to set OFFBOARD if not set, arms if not armed,
        then publishes the velocity from the callback.
        """
        while not rospy.is_shutdown():
            # Try to set OFFBOARD mode
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req > rospy.Duration(5.0)):
                if self.set_mode_client.call(self.offb_set_mode).mode_sent:
                    rospy.loginfo("Offboard enabled")
                self.last_req = rospy.Time.now()

            # Try to arm
            elif not self.current_state.armed and (rospy.Time.now() - self.last_req > rospy.Duration(5.0)):
                if self.arming_client.call(self.arm_cmd).success:
                    rospy.loginfo("Vehicle armed")
                self.last_req = rospy.Time.now()

            # Keep publishing the latest velocity setpoint
            self.local_pos_pub.publish(self.pose_msg)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        node = VisualServoingPIDNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
