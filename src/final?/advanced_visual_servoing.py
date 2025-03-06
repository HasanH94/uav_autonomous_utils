#!/usr/bin/env python3
import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class CombinedSearchAndPIDNode:
    def __init__(self):
        rospy.init_node("combined_search_and_pid_node", anonymous=True)

        # -----------------------------
        # Parameters / Settings
        # -----------------------------
        self.search_rotation_speed = rospy.get_param("~search_rotation_speed", 0.5)
        self.search_timeout = rospy.get_param("~search_timeout", 30.0)    # seconds
        self.pid_lost_object_time = rospy.get_param("~pid_lost_object_time", 5.0)  # seconds
        self.stop_distance = rospy.get_param("~stop_distance", 1.2)      # meters

        # Gains for PID (tune these as needed)
        # Note: Our mapping is:
        #   - forward velocity (body x) comes from err_z (marker distance error)
        #   - lateral velocity (body y) comes from err_x (marker lateral error)
        #   - vertical velocity (body z) comes from err_y (marker vertical error), but inverted.
        self.k_x   = rospy.get_param("~k_x",   0.22)   # for err_z (forward)
        self.k_y   = rospy.get_param("~k_y",   0.35)   # for err_x (lateral)
        self.k_z   = rospy.get_param("~k_z",   0.3)    # for err_y (vertical)
        self.k_yaw = rospy.get_param("~k_yaw", 0.1)    # for yaw error

        self.kd_x   = rospy.get_param("~kd_x",   0.0)
        self.kd_y   = rospy.get_param("~kd_y",   0.005)
        self.kd_z   = rospy.get_param("~kd_z",   0.0)
        self.kd_yaw = rospy.get_param("~kd_yaw", 0.03)

        self.yaw_vel_clamp = 0.2  # Clamp yaw velocity

        # -----------------------------
        # MAVROS and FCU Setup
        # -----------------------------
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/arming")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        # -----------------------------
        # Pose & Yaw
        # -----------------------------
        self.yaw = 0.0
        self.current_position = None
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        # -----------------------------
        # Marker and Visual Error Data
        # -----------------------------
        self.marker_detected = False
        self.marker_sub = rospy.Subscriber("/marker_detection_status", Bool, self.marker_detected_cb)
        self.error_sub = rospy.Subscriber("/visual_errors", Twist, self.visual_error_cb)
        self.latest_error = None  # Expected to be in body frame: err_x (lateral), err_y (vertical), err_z (forward), err_yaw

        # -----------------------------
        # Publishers
        # -----------------------------
        # For velocity commands, we publish an unstamped Twist.
        self.cmd_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.timeout_pub = rospy.Publisher("/search_timeout", Bool, queue_size=10)
        self.pose_setpoint_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

        # -----------------------------
        # Internal State
        # -----------------------------
        self.mode = "search"  # Modes: "search" or "pid"
        self.search_start_time = None
        self.pid_lost_start = None

        # For PID integration and derivative calculations
        self.integral_x = self.integral_y = self.integral_z = self.integral_yaw = 0.0
        self.prev_x = self.prev_y = self.prev_z = self.prev_yaw_err = 0.0
        self.dt = 1.0 / 20.0  # 20 Hz
        self.rate = rospy.Rate(20)
        self.last_req = rospy.Time.now()

        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected.")

        # Send some zero setpoints before OFFBOARD mode
        zero_twist = Twist()
        for _ in range(50):
            self.cmd_pub.publish(zero_twist)
            self.rate.sleep()

        self.offb_req = SetModeRequest()
        self.offb_req.custom_mode = "OFFBOARD"
        self.arm_req = CommandBoolRequest()
        self.arm_req.value = True

    # -----------------------------
    # MAVROS Callbacks
    # -----------------------------
    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, pose_msg):
        q = pose_msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw
        self.current_position = pose_msg.pose.position

    def marker_detected_cb(self, msg):
        self.marker_detected = msg.data

    def visual_error_cb(self, error_msg):
        self.latest_error = error_msg

    # -----------------------------
    # Main Run Loop
    # -----------------------------
    def run(self):
        rospy.loginfo("CombinedSearchAndPIDNode is running...")
        self.search_start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            self.handle_flight_mode_and_arming()

            if self.mode == "search":
                self.search_mode_step()
            elif self.mode == "pid":
                self.pid_mode_step()

            self.rate.sleep()

    def handle_flight_mode_and_arming(self):
        now = rospy.Time.now()
        if (self.current_state.mode != "OFFBOARD") and (now - self.last_req > rospy.Duration(5.0)):
            resp1 = self.set_mode_client(self.offb_req)
            if resp1.mode_sent:
                rospy.loginfo("Offboard enabled")
            self.last_req = now
        elif (not self.current_state.armed) and (now - self.last_req > rospy.Duration(5.0)):
            arm_resp = self.arming_client(self.arm_req)
            if arm_resp.success:
                rospy.loginfo("Vehicle armed")
            self.last_req = now

    # -----------------------------
    # Search Mode: Rotate until marker is detected or timeout
    # -----------------------------
    def search_mode_step(self):
        elapsed = rospy.Time.now().to_sec() - self.search_start_time
        if self.marker_detected:
            rospy.loginfo("Marker found! Switching to PID mode.")
            self.switch_to_pid()
            return

        if elapsed > self.search_timeout:
            rospy.logwarn(f"Search timed out after {self.search_timeout}s, giving up.")
            self.timeout_pub.publish(True)
            twist = Twist()
            for _ in range(10):
                self.cmd_pub.publish(twist)
            rospy.signal_shutdown("Search timeout reached.")
            return

        twist = Twist()
        twist.angular.z = self.search_rotation_speed
        self.cmd_pub.publish(twist)

    def switch_to_pid(self):
        self.mode = "pid"
        self.integral_x = self.integral_y = self.integral_z = self.integral_yaw = 0.0
        self.prev_x = self.prev_y = self.prev_z = self.prev_yaw_err = 0.0
        self.pid_lost_start = None
        zero_twist = Twist()
        for _ in range(20):
            self.cmd_pub.publish(zero_twist)
        rospy.sleep(2)

    # -----------------------------
    # PID Mode: Process visual errors, transform commands, and publish Twist
    # -----------------------------
    def pid_mode_step(self):
        if not self.marker_detected:
            if self.pid_lost_start is None:
                self.pid_lost_start = rospy.Time.now().to_sec()
            else:
                lost_elapsed = rospy.Time.now().to_sec() - self.pid_lost_start
                if lost_elapsed > self.pid_lost_object_time:
                    rospy.logwarn("Marker lost for 5s -> switching back to SEARCH mode.")
                    self.switch_to_search()
                    return
        else:
            self.pid_lost_start = None

        if self.latest_error is None:
            self.cmd_pub.publish(Twist())
            return

        # Extract visual errors (assumed in body frame)
        err_x   = self.latest_error.linear.x  # lateral error
        err_y   = self.latest_error.linear.y  # vertical error
        err_z   = self.latest_error.linear.z  # forward error
        err_yaw = self.latest_error.angular.z # yaw error

        distance = math.sqrt(err_x**2 + err_y**2 + err_z**2)
        if distance < self.stop_distance:
            rospy.loginfo(f"Within stop distance ({self.stop_distance}m). Holding position.")
            self.publish_position_hold()
            return

        # Update PID integrals
        self.integral_x   += err_x * self.dt
        self.integral_y   += err_y * self.dt
        self.integral_z   += err_z * self.dt
        self.integral_yaw += err_yaw * self.dt

        # Derivatives
        dx   = (err_x   - self.prev_x) / self.dt
        dy   = (err_y   - self.prev_y) / self.dt
        dz   = (err_z   - self.prev_z) / self.dt
        dyaw = (err_yaw - self.prev_yaw_err) / self.dt

        # Compute body-frame PID commands:
        #   - Forward (body x) from err_z
        #   - Lateral (body y) from err_x
        #   - Vertical (body z) from err_y, but inverted so positive err_y (drone above marker) yields downward motion
        body_x_vel = (self.k_x * err_z) + (self.kd_x * dz)
        body_y_vel = (self.k_y * err_x) + (self.kd_y * dx)
        body_z_vel = (self.k_z * err_y) + (self.kd_z * dy)
        body_yaw_vel = (self.k_yaw * err_yaw) + (self.kd_yaw * dyaw)
        body_yaw_vel = max(min(body_yaw_vel, self.yaw_vel_clamp), -self.yaw_vel_clamp)

        # In our body frame, the vertical command should be inverted:
        body_z_vel = -body_z_vel

        # Transform body-frame velocities to global frame using the current yaw.
        global_vel_x = body_x_vel * math.cos(self.yaw) - body_y_vel * math.sin(self.yaw)
        global_vel_y = body_x_vel * math.sin(self.yaw) + body_y_vel * math.cos(self.yaw)
        global_vel_z = body_z_vel  # vertical velocity remains the same

        rospy.loginfo("PID errors: err_x=%.2f, err_y=%.2f, err_z=%.2f, err_yaw=%.2f",
                      err_x, err_y, err_z, err_yaw)
        rospy.loginfo("Body-frame velocities: [%.2f, %.2f, %.2f], Yaw rate: %.2f",
                      body_x_vel, body_y_vel, body_z_vel, body_yaw_vel)
        rospy.loginfo("Global velocities: [%.2f, %.2f, %.2f]",
                      global_vel_x, global_vel_y, global_vel_z)

        # Create Twist message (global velocities)
        twist = Twist()
        twist.linear.x  = global_vel_x
        twist.linear.y  = global_vel_y
        twist.linear.z  = global_vel_z
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = body_yaw_vel  # Yaw rate remains in body frame

        self.cmd_pub.publish(twist)

        # Update previous errors for derivative calculation
        self.prev_x = err_x
        self.prev_y = err_y
        self.prev_z = err_z
        self.prev_yaw_err = err_yaw

    def publish_position_hold(self):
        if self.current_position is None:
            rospy.logwarn("Current position not available; cannot hold position.")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"  # Adjust as needed
        pose_msg.pose.position.x = self.current_position.x
        pose_msg.pose.position.y = self.current_position.y
        pose_msg.pose.position.z = self.current_position.z

        quat = quaternion_from_euler(0.0, 0.0, self.yaw)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_setpoint_pub.publish(pose_msg)
        rospy.loginfo("Published position setpoint for hold.")

        try:
            set_mode_req = SetModeRequest()
            set_mode_req.custom_mode = "AUTO.LOITER"
            resp = self.set_mode_client(set_mode_req)
            if resp.mode_sent:
                rospy.loginfo("Switched to AUTO.LOITER mode.")
            else:
                rospy.logwarn("Failed to switch to AUTO.LOITER mode.")
        except rospy.ServiceException as e:
            rospy.logerr(f"SetMode service call failed: {e}")

        rospy.loginfo("Shutting down node.")
        rospy.signal_shutdown("Stop distance reached; switching to LOITER.")

    def switch_to_search(self):
        rospy.loginfo("Switching to SEARCH mode.")
        self.mode = "search"
        self.search_start_time = rospy.Time.now().to_sec()
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.5)

if __name__ == "__main__":
    try:
        node = CombinedSearchAndPIDNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
