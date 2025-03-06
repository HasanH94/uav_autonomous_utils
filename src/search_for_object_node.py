#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, SetModeRequest

class SearchForObjectNode:
    def __init__(self):
        rospy.init_node("search_for_object_node", anonymous=True)

        # Parameters
        self.rotation_speed = rospy.get_param("~rotation_speed", 0.5)
        # Which mode to switch to on shutdown (e.g. "LOITER" or "POSCTL" for PX4, "AUTO.LOITER" for ArduPilot).
        self.hover_mode = rospy.get_param("~hover_mode", "AUTO.LOITER")

        # Rate at which we publish velocity commands
        self.rate = rospy.Rate(10)  # 10 Hz

        # State subscriber
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.current_state = State()

        # Velocity publisher
        self.velocity_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped",
                                            Twist, queue_size=10)

        # Set mode service
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # Register on_shutdown callback
        rospy.on_shutdown(self.shutdown_cb)

        rospy.loginfo("SearchForObjectNode initialized.")

    def state_cb(self, msg):
        """Keep track of the current FCU state."""
        self.current_state = msg

    def wait_for_connection(self):
        """Block until connected to FCU."""
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected.")

    def set_offboard_mode(self):
        """Switch the drone to OFFBOARD mode, publishing minimal velocity setpoints until it switches."""
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        last_req = rospy.Time.now()

        while not rospy.is_shutdown() and self.current_state.mode != "OFFBOARD":
            # Try setting OFFBOARD every 2 seconds
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                resp = self.set_mode_client(offb_set_mode)
                if resp.mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled.")
                else:
                    rospy.logwarn("Failed to set OFFBOARD mode.")
                last_req = rospy.Time.now()

            # Publish minimal setpoint so FCU accepts OFFBOARD
            self.publish_velocity_setpoint(0.0)
            self.rate.sleep()

        rospy.loginfo("Drone is in OFFBOARD mode. Starting spin...")

    def publish_velocity_setpoint(self, z_rotation):
        """Publish a velocity command with zero linear motion and z_rotation in angular.z."""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.z = z_rotation
        self.velocity_pub.publish(twist_msg)

    def spin_in_place(self):
        """
        Main loop: keep rotating until ROS is shutdown (or an external process kills this node).
        """
        while not rospy.is_shutdown():
            self.publish_velocity_setpoint(self.rotation_speed)
            self.rate.sleep()

    def run(self):
        """Orchestration: connect, set OFFBOARD, and begin rotating."""
        self.wait_for_connection()
        self.set_offboard_mode()
        self.spin_in_place()

    def shutdown_cb(self):
        """
        Called when this node is killed (Ctrl-C or otherwise).
        We'll switch the drone to LOITER (or user-specified mode) and publish zero velocity briefly.
        """
        try:
            rospy.loginfo("Node is shutting down. Switching to hover mode.")
            hold_mode = SetModeRequest()
            hold_mode.custom_mode = self.hover_mode
            resp = self.set_mode_client.call(hold_mode)
            if resp.mode_sent:
                rospy.loginfo(f"Switched to {self.hover_mode} mode successfully.")
            else:
                rospy.logwarn(f"Failed to switch to {self.hover_mode} mode.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        # Publish zero velocity commands for 1 second so the autopilot stops rotating
        t_end = rospy.Time.now() + rospy.Duration(1.0)
        while rospy.Time.now() < t_end and not rospy.is_shutdown():
            twist_msg = Twist()  # all zeros
            self.velocity_pub.publish(twist_msg)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = SearchForObjectNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
