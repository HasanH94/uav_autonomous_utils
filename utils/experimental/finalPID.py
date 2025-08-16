#!/usr/bin/env python3
import rospy
import math
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
        self.search_timeout = rospy.get_param("~search_timeout", 30.0)    # 30s
        self.pid_lost_object_time = rospy.get_param("~pid_lost_object_time", 5.0)  # 5s
        self.stop_distance = rospy.get_param("~stop_distance", 1.2)      # 1.5m

        # Gains for PID
        self.k_x   = rospy.get_param("~k_x",   0.22)
        self.k_y   = rospy.get_param("~k_y",   0.35)
        self.k_z   = rospy.get_param("~k_z",   0.3)
        self.k_yaw = rospy.get_param("~k_yaw", 0.1)

        self.kd_x   = rospy.get_param("~kd_x",   0.0)
        self.kd_y   = rospy.get_param("~kd_y",   0.005)
        self.kd_z   = rospy.get_param("~kd_z",   0.0)
        self.kd_yaw = rospy.get_param("~kd_yaw", 0.03)

        # We'll clamp yaw velocity to ±0.2 rad/s
        self.yaw_vel_clamp = 0.2

        # MAVROS and FCU
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/arming")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        # Pose (for yaw and position)
        self.yaw = 0.0
        self.current_position = None  # Initialize current_position
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        # Marker detection
        self.marker_detected = False
        self.marker_sub = rospy.Subscriber("/marker_detection_status", Bool, self.marker_detected_cb)

        # Visual errors for PID
        self.error_sub = rospy.Subscriber("/visual_errors", Twist, self.visual_error_cb)
        self.latest_error = None  # store the latest Twist from /visual_errors

        # Velocity publisher
        self.cmd_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped",
                                       Twist, queue_size=10)

        # Publisher for search timeout
        self.timeout_pub = rospy.Publisher("/search_timeout", Bool, queue_size=10)  # Added Publisher

        # Publisher for position setpoints
        self.pose_setpoint_pub = rospy.Publisher("/mavros/setpoint_position/local",
                                                 PoseStamped, queue_size=10)  # Added Publisher

        # Internal state: "search" or "pid"
        self.mode = "search"

        # Timers
        self.search_start_time = None    # for 30s search timeout
        self.pid_lost_start = None       # track how long we've lost marker in PID

        # PID integrals, derivatives
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0
        self.integral_yaw = 0.0

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
        self.prev_yaw_err = 0.0

        self.dt = 1.0 / 20.0  # 20 Hz
        self.rate = rospy.Rate(20)

        self.last_req = rospy.Time.now()

        # Wait for FCU connection
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected.")

        # Send some zero setpoints before OFFBOARD
        zero_twist = Twist()
        for _ in range(50):
            self.cmd_pub.publish(zero_twist)
            self.rate.sleep()

        # Attempt to set OFFBOARD + arm
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
        # Extract yaw
        q = pose_msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw
        # Store current position
        self.current_position = pose_msg.pose.position

    def marker_detected_cb(self, msg):
        self.marker_detected = msg.data

    def visual_error_cb(self, error_msg):
        self.latest_error = error_msg

    # -----------------------------
    # Main run loop
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

    # -----------------------------
    # OFFBOARD + Arming
    # -----------------------------
    def handle_flight_mode_and_arming(self):
        # Try OFFBOARD
        now = rospy.Time.now()
        if (self.current_state.mode != "OFFBOARD") and (now - self.last_req > rospy.Duration(5.0)):
            resp1 = self.set_mode_client(self.offb_req)
            if resp1.mode_sent:
                rospy.loginfo("Offboard enabled")
            self.last_req = now

        # Try arming
        elif (not self.current_state.armed) and (now - self.last_req > rospy.Duration(5.0)):
            arm_resp = self.arming_client(self.arm_req)
            if arm_resp.success:
                rospy.loginfo("Vehicle armed")
            self.last_req = now

    # -----------------------------
    # Search Mode
    # -----------------------------
    def search_mode_step(self):
        """
        If marker_detected -> switch to PID
        If 30 seconds pass -> publish /search_timeout=True
        Otherwise, spin in place.
        """
        elapsed = rospy.Time.now().to_sec() - self.search_start_time

        # Check if we found the marker
        if self.marker_detected:
            rospy.loginfo("Marker found! Switching to PID mode.")
            self.switch_to_pid()
            return

        # Check for 30s timeout
        if elapsed > self.search_timeout:
            rospy.logwarn(f"Search timed out after {self.search_timeout}s, giving up.")
            # Publish /search_timeout=True
            self.timeout_pub.publish(True)  # Publishing True to indicate timeout
            # Optionally, you can reset the flag after publishing
            # rospy.sleep(0.1)
            # self.timeout_pub.publish(False)
            # Or leave it as is, assuming the state machine will handle the transition
            # Stop rotating
            twist = Twist()
            for _ in range(10):
                self.cmd_pub.publish(twist)

            rospy.signal_shutdown("Search timeout reached.")
            return  # or continue if you want to keep spinning anyway

        # Otherwise, spin in place
        twist = Twist()
        twist.angular.z = self.search_rotation_speed
        self.cmd_pub.publish(twist)

    def switch_to_pid(self):
        self.mode = "pid"
        # Reset integrals, derivatives, so we don't carry stale errors from last time
        self.integral_x = self.integral_y = self.integral_z = self.integral_yaw = 0.0
        self.prev_x = self.prev_y = self.prev_z = self.prev_yaw_err = 0.0
        # Reset lost marker timer
        self.pid_lost_start = None
        zeroTwist  = Twist()
        for i in range(20):
            self.cmd_pub.publish(zeroTwist)
        rospy.sleep(2)

    # -----------------------------
    # PID Mode
    # -----------------------------
    def pid_mode_step(self):
        """
        If marker is lost for 5 continuous seconds -> switch to search
        If within stop_distance, publish position setpoint to hold position
        Otherwise, do the PID logic
        """
        # Check marker detection
        if not self.marker_detected:
            # If we don't already have a start time, record it
            if self.pid_lost_start is None:
                self.pid_lost_start = rospy.Time.now().to_sec()
            else:
                lost_elapsed = rospy.Time.now().to_sec() - self.pid_lost_start
                if lost_elapsed > self.pid_lost_object_time:
                    rospy.logwarn("Marker lost for 5s -> switching back to SEARCH.")
                    self.switch_to_search()
                    return
        else:
            # We see the marker -> reset
            self.pid_lost_start = None

        # If we have no /visual_errors yet, just publish zero
        if self.latest_error is None:
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        # Extract errors
        err_x = self.latest_error.linear.x
        err_y = self.latest_error.linear.y
        err_z = self.latest_error.linear.z
        err_yaw = self.latest_error.angular.z

        distance = math.sqrt(err_x**2 + err_y**2 + err_z**2)

        # ---------------------------
        # Stop condition with Position Hold
        # ---------------------------
        if distance < self.stop_distance:
            rospy.loginfo(f"Within stop distance ({self.stop_distance}m). Holding position.")
            self.publish_position_hold()
            return

        # ---------------------------
        # PID logic
        # ---------------------------
        # Integrals
        self.integral_x   += err_x * self.dt
        self.integral_y   += err_y * self.dt
        self.integral_z   += err_z * self.dt
        self.integral_yaw += err_yaw * self.dt

        # Derivatives
        dx   = (err_x   - self.prev_x) / self.dt
        dy   = (err_y   - self.prev_y) / self.dt
        dz   = (err_z   - self.prev_z) / self.dt
        dyaw = (err_yaw - self.prev_yaw_err) / self.dt

        # body_x_vel depends on err_z, etc. (like your code)
        body_x_vel = (self.k_x * err_z) + (self.kd_x * dz)
        body_y_vel = (self.k_y * err_x) + (self.kd_y * dx)
        body_z_vel = (self.k_z * err_y) + (self.kd_z * dy)

        # Yaw
        body_yaw_vel = (self.k_yaw * err_yaw) + (self.kd_yaw * dyaw)

        # Convert body velocities to global using current yaw
        global_vel_x = body_x_vel * math.cos(self.yaw) - body_y_vel * math.sin(self.yaw)
        global_vel_y = body_x_vel * math.sin(self.yaw) + body_y_vel * math.cos(self.yaw)
        global_vel_z = body_z_vel

        # Clamp yaw
        body_yaw_vel = max(min(body_yaw_vel,  self.yaw_vel_clamp),
                           -self.yaw_vel_clamp)

        # Construct twist
        twist = Twist()
        twist.linear.x  = global_vel_x
        twist.linear.y  = global_vel_y
        twist.linear.z  = global_vel_z
        twist.angular.z = body_yaw_vel
        self.cmd_pub.publish(twist)

        # Update prev
        self.prev_x = err_x
        self.prev_y = err_y
        self.prev_z = err_z
        self.prev_yaw_err = err_yaw

    def publish_position_hold(self):
        """
        Publish the current position as a setpoint to hold the drone's position.
        After publishing, switch to AUTO.LOITER mode and shutdown the node.
        """
        if self.current_position is None:
            rospy.logwarn("Current position is not available. Cannot publish position hold.")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"  # Ensure this matches your flight controller's frame

        # Set position
        pose_msg.pose.position.x = self.current_position.x
        pose_msg.pose.position.y = self.current_position.y
        pose_msg.pose.position.z = self.current_position.z

        # Set orientation (keep current yaw)
        quat = quaternion_from_euler(0.0, 0.0, self.yaw)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        # Publish the position setpoint
        self.pose_setpoint_pub.publish(pose_msg)
        rospy.loginfo("Published position setpoint to hold position.")

        # Switch to AUTO.LOITER mode
        try:
            set_mode = SetModeRequest()
            set_mode.custom_mode = "AUTO.LOITER"
            resp = self.set_mode_client(set_mode)
            if resp.mode_sent:
                rospy.loginfo("Switched to AUTO.LOITER mode.")
            else:
                rospy.logwarn("Failed to switch to AUTO.LOITER mode.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        # Shutdown the node
        rospy.loginfo("Shutting down CombinedSearchAndPIDNode.")
        rospy.signal_shutdown("Reached stop distance, switching to LOITER mode.")

    def switch_to_search(self):
        rospy.loginfo("Switching to SEARCH mode.")
        self.mode = "search"
        self.search_start_time = rospy.Time.now().to_sec()

        # Stop the drone briefly
        twist = Twist()
        self.cmd_pub.publish(twist)
        rospy.sleep(0.5)

if __name__ == "__main__":
    try:
        node = CombinedSearchAndPIDNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
