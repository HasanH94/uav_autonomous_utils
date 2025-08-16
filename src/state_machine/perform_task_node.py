#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State
import time

class PerformTaskNode:
    def __init__(self):
        rospy.init_node("perform_task_node", anonymous=True)

        # Publisher for the task done signal
        self.task_done_pub = rospy.Publisher('/drone_events/task_done', Bool, queue_size=10)

        # State subscriber to check current mode
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        # Set mode service client
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        rospy.loginfo("PerformTaskNode initialized.")

    def state_cb(self, msg):
        """Keep track of the current FCU state."""
        self.current_state = msg

    def set_loiter_mode(self):
        """Switch the drone to AUTO.LOITER mode."""
        loiter_set_mode = SetModeRequest()
        loiter_set_mode.custom_mode = 'AUTO.LOITER'
        last_req = rospy.Time.now()

        rospy.loginfo("Attempting to set AUTO.LOITER mode...")
        while not rospy.is_shutdown() and self.current_state.mode != "AUTO.LOITER":
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                try:
                    resp = self.set_mode_client(loiter_set_mode)
                    if resp.mode_sent:
                        rospy.loginfo("AUTO.LOITER mode change request sent.")
                    else:
                        rospy.logwarn("Failed to send AUTO.LOITER mode change request.")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call to set_mode failed: {e}")
                last_req = rospy.Time.now()
            
            rospy.sleep(0.1)

        if self.current_state.mode == "AUTO.LOITER":
            rospy.loginfo("Drone is in AUTO.LOITER mode.")
            return True
        else:
            rospy.logwarn("Failed to switch to AUTO.LOITER mode.")
            return False

    def run(self):
        """Orchestration: set LOITER, wait, and publish task_done."""
        # Wait for FCU connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.loginfo("Waiting for FCU connection...")
            rospy.sleep(1)
        rospy.loginfo("FCU connected.")

        if self.set_loiter_mode():
            rospy.loginfo("Mode set to AUTO.LOITER. Waiting for 5 seconds...")
            time.sleep(5.0)

            rospy.loginfo("Wait complete. Publishing task_done signal.")
            self.task_done_pub.publish(Bool(data=True))
            rospy.sleep(1.0)  # Give time for the message to be published
        else:
            rospy.logerr("Failed to set AUTO.LOITER mode. Aborting.")

        rospy.loginfo("Task finished. Node will remain active until shut down by the state manager.")
        rospy.spin() # Keep the node alive until it's shut down externally


if __name__ == "__main__":
    try:
        node = PerformTaskNode()
        node.run()
    except rospy.ROSInterruptException:
        pass