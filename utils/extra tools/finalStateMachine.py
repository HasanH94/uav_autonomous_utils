#!/usr/bin/env python3
import rospy
import roslaunch
import subprocess
from transitions import Machine
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Header
import math
import signal
import sys
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from tf.transformations import quaternion_from_euler


class StateManager:
    """
    This class is responsible for launching/stopping the actual ROS processes
    (like local_node.launch, search_for_object_node.py, visual_servoing_node.py, etc.).
    """
    def __init__(self):
        self.current_process = None

    def launch_file(self, package, launch_file, publish_topic=None, *args):
        """Launch a .launch file."""
        rospy.loginfo(f"[StateManager] Attempting to launch file: {package}/{launch_file}")
        self.stop_current_process()
        try:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]
            self.current_process = roslaunch.parent.ROSLaunchParent(
                uuid, [launch_path],
                sigint_timeout=1.0, sigterm_timeout=1.0
            )
            self.current_process.start()
            rospy.loginfo(f"[StateManager] Launched: {package}/{launch_file}")
            rospy.sleep(1)

            # Optionally publish goal if 3 numeric args were given
            if len(args) == 3 and all(isinstance(arg, (int, float)) for arg in args):
                self.publish_goal(*args, publish_topic)
            else:
                rospy.logwarn("[StateManager] Invalid arguments for publish_goal. Expected three numerical values.")
            rospy.sleep(1)
        except Exception as e:
            rospy.logerr(f"[StateManager] Failed to launch {package}/{launch_file}: {e}")

    def run_node(self, package, executable):
        """
        Use rosrun to start a single node (Python script or otherwise).
        For example: run_node('my_package', 'search_for_object_node.py').
        """
        rospy.loginfo(f"[StateManager] Attempting to run node: {package}/{executable}")
        self.stop_current_process()
        try:
            command = f"rosrun {package} {executable}"
            self.current_process = subprocess.Popen(command, shell=True)
            rospy.loginfo(f"[StateManager] Started node via: {command}")
        except Exception as e:
            rospy.logerr(f"[StateManager] Failed to run node {package}/{executable}: {e}")

    def stop_current_process(self):
        """Stop any currently running process or launch file."""
        if self.current_process:
            rospy.loginfo(f"[StateManager] Stopping current process: {self.current_process}")
            try:
                if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
                    self.current_process.shutdown()
                    rospy.loginfo("[StateManager] roslaunch parent shutdown complete.")
                else:
                    self.current_process.terminate()
                    self.current_process.wait()
                    rospy.loginfo("[StateManager] Terminated subprocess.")
                rospy.loginfo("[StateManager] Stopped current process.")
            except Exception as e:
                rospy.logerr(f"[StateManager] Error stopping process: {e}")
            finally:
                self.current_process = None
        else:
            rospy.loginfo("[StateManager] No current process to stop.")

    def publish_goal(self, x, y, z, publish_topic="/move_base_simple/goal"):
        """Publish a PoseStamped message to /move_base_simple/goal by default."""
        goal_pub = rospy.Publisher(publish_topic, PoseStamped, queue_size=10)
        rospy.sleep(1)

        # Wait until at least one subscriber is connected, or 5s
        rate = rospy.Rate(10)
        timeout_seconds = 5
        start_time = rospy.Time.now()
        while goal_pub.get_num_connections() < 1 and not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout_seconds:
                rospy.logwarn("No subscribers connected after 5s, proceeding anyway.")
                break
            rate.sleep()

        # Create and populate the PoseStamped message
        goal = PoseStamped()
        goal.header = Header()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        # Orientation: identity quaternion
        # goal.pose.orientation.x = 0.0
        # goal.pose.orientation.y = 0.0
        # goal.pose.orientation.z = 0.0
        # goal.pose.orientation.w = 1.0

        # Publish the goal
        goal_pub.publish(goal)
        rospy.loginfo(f"[StateManager] Published goal to {publish_topic}: x={x}, y={y}, z={z}")


class DroneStateMachine:
    # Define states, including the new "searching" state
    states = [
        {'name': 'gps_navigation', 'on_enter': 'logGpsStart'},
        {'name': 'searching', 'on_enter': 'logSearchStart'},
        {'name': 'visual_servoing', 'on_enter': 'logVisualServoingStart'},
        {'name': 'performing_task', 'on_enter': 'logTaskStart'},
    ]

    def __init__(self, goal_points):
        self.manager = StateManager()
        self.goal_points = goal_points
        self.counter = 0
        self.current_position = None
        self.goal_position = None
        self.threshold = 0.2

        # Marker detection
        self.marker_detected = False

        # Set up MAVROS services for mode and arming
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd_arming')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd_arming', CommandBool)
        rospy.loginfo("[DroneStateMachine] Service proxies for set_mode and arming initialized.")

        # Build the transitions
        self.machine = Machine(
            model=self,
            states=DroneStateMachine.states,
            initial='gps_navigation',
            auto_transitions=False,
            send_event=True
        )

        # GPS -> Searching -> Visual -> Task -> GPS
        self.machine.add_transition(
            trigger='REACHED_GOAL_POINT',
            source='gps_navigation',
            dest='searching'
        )
        self.machine.add_transition(
            trigger='FOUND_MARKER',
            source='searching',
            dest='visual_servoing'
        )
        self.machine.add_transition(
            trigger='REACHED_DESIRED_POSE',
            source='visual_servoing',
            dest='performing_task'
        )
        self.machine.add_transition(
            trigger='TASK_DONE',
            source='performing_task',
            dest='gps_navigation'
        )

        # Additional transitions if you still want timeouts:
        # searching -> gps_navigation if TIMEOUT
        self.machine.add_transition(
            trigger='TIMEOUT',
            source='searching',
            dest='gps_navigation'
        )
        # visual_servoing -> gps_navigation if TIMEOUT or marker lost
        # (this depends on your logic preference)
        # self.machine.add_transition(
        #     trigger='TIMEOUT',
        #     source='visual_servoing',
        #     dest='gps_navigation'
        # )

        # Start in GPS Navigation
        self.logGpsStart()

        # ROS Subscribers
        self._init_subscribers()

    def _init_subscribers(self):
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/marker_detection_status', Bool, self.marker_detection_callback)
        rospy.Subscriber('/visual_errors', Twist, self.visual_error_callback)
        rospy.Subscriber('/search_timeout', Bool, self.search_timeout_callback)
        rospy.Subscriber('/task_done', Bool, self.task_done_callback)

        # Subscribe to a "marker_lost" topic published by your pure PID node if you want
        rospy.Subscriber('/marker_lost', Bool, self.marker_lost_callback)

        rospy.loginfo("[DroneStateMachine] Subscribers initialized.")

    # ---------------
    # Callbacks
    # ---------------
    def marker_detection_callback(self, msg):
        """If marker is detected and we are in 'searching', trigger FOUND_MARKER."""
        self.marker_detected = msg.data
        if self.marker_detected and self.state == 'searching':
            rospy.loginfo("[DroneStateMachine] Marker detected! Switching to visual_servoing.")
            self.FOUND_MARKER()

    def marker_lost_callback(self, msg):
        """If marker is lost in 'visual_servoing', we might want to return to searching or gps."""
        if msg.data and self.state == 'visual_servoing':
            rospy.logwarn("[DroneStateMachine] Marker lost -> TIMEOUT. Returning to gps_navigation.")
            self.TIMEOUT()  # or some custom trigger to go back to searching

    def search_timeout_callback(self, msg):
        """
        If your search_for_object_node somehow publishes /search_timeout=True,
        then you can transition back to GPS or do something else.
        """
        if msg.data and self.state == 'searching':
            rospy.loginfo("[DroneStateMachine] Received /search_timeout=True -> TIMEOUT.")
            self.TIMEOUT()

    def task_done_callback(self, msg):
        """If True and we are in performing_task, then go back to gps_navigation."""
        if msg.data and self.state == 'performing_task':
            rospy.loginfo("[DroneStateMachine] Received /task_done=True, triggering TASK_DONE.")
            self.TASK_DONE()

    def goal_callback(self, msg):
        """Update the goal_position from /goal_position (MarkerArray)."""
        if not msg.markers:
            rospy.logwarn("[DroneStateMachine] Received empty MarkerArray on /goal_position.")
            return

        marker = msg.markers[0]
        self.goal_position = marker.pose.position
        rospy.loginfo(
            f"[DroneStateMachine] New goal: x={self.goal_position.x}, y={self.goal_position.y}, z={self.goal_position.z}"
        )

    def position_callback(self, msg):
        """Check if we have reached the goal in gps_navigation."""
        self.current_position = msg.pose.position
        if self.goal_position is None:
            return

        distance = self.calculate_distance(self.current_position, self.goal_position)
        if distance <= self.threshold and self.state == 'gps_navigation':
            rospy.loginfo("[DroneStateMachine] Drone reached the goal position.")
            self.goal_position = None
            self.counter += 1

            # If there are more goals left, trigger REACHED_GOAL_POINT -> searching
            if len(self.goal_points) >= self.counter:
                rospy.loginfo("[DroneStateMachine] Triggering REACHED_GOAL_POINT -> searching.")
                self.REACHED_GOAL_POINT()
            else:
                # No more goals => Go home & land
                pass

    def visual_error_callback(self, msg):
        """
        If in visual_servoing, check if error norm < 0.2 => REACHED_DESIRED_POSE.
        (Same as before, or adapt as needed.)
        """
        if self.state == 'visual_servoing':
            x_err = msg.linear.x
            y_err = msg.linear.y
            z_err = msg.linear.z
            error_norm = math.sqrt(x_err**2 + y_err**2 + z_err**2)

            # If we consider ~1.2 the desired distance, like in old code:
            if abs(1.2 - error_norm) < 0.2:
                rospy.loginfo("[DroneStateMachine] Visual errors < 0.2 => REACHED_DESIRED_POSE()")
                self.REACHED_DESIRED_POSE()

    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2)

    # ---------------
    # On-enter Actions
    # ---------------
    def logGpsStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered GPS Navigation State")
        self.manager.stop_current_process()
        self.marker_detected = False
        # Attempt to launch local planner with the next goal
        try:
            current_goal = self.goal_points[self.counter]
            self.manager.launch_file('local_planner', 'local_node.launch',
                                     "/move_base_simple/goal",
                                     *current_goal)
            rospy.loginfo("[DroneStateMachine] GPS Navigation Node Launched")

            rospy.sleep(2)  # Wait for setpoints to come up
            if self.set_offboard_mode():
                rospy.sleep(1)
                if self.arm_drone():
                    rospy.loginfo("[DroneStateMachine] Drone armed successfully (OFFBOARD).")
                else:
                    rospy.logwarn("[DroneStateMachine] Drone failed to arm.")
            else:
                rospy.logwarn("[DroneStateMachine] OFFBOARD mode not set.")
        except IndexError:
            # No more goals => go home
            home_position = PoseStamped()
            home_position.pose.position.x = 0
            home_position.pose.position.y = 0
            home_position.pose.position.z = 1
            rospy.logerr("No more goals. Moving to home position and landing.")

            self.manager.launch_file('local_planner', 'local_node.launch',
                                     "/move_base_simple/goal",
                                     home_position.pose.position.x,
                                     home_position.pose.position.y,
                                     home_position.pose.position.z)
            self.set_offboard_mode()
            rospy.sleep(2)

            while self.current_position is not None and \
                  self.calculate_distance(self.current_position, home_position.pose.position) > self.threshold \
                  and not rospy.is_shutdown():
                rospy.sleep(1)

            # Land
            land_req = SetModeRequest()
            land_req.custom_mode = "AUTO.LAND"
            try:
                resp = self.set_mode_client(land_req)
                if resp.mode_sent:
                    rospy.loginfo("[DroneStateMachine] Landing mode enabled.")
                else:
                    rospy.logwarn("[DroneStateMachine] Failed to set LAND mode.")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed while setting LAND mode: {e}")

            rospy.signal_shutdown("No more goals -> Landed at home.")

    def logSearchStart(self, event=None):
        """
        On entering 'searching' state, we'll run the search_for_object_node.py
        which spins in place until we kill it or switch states.
        """
        rospy.loginfo("[DroneStateMachine] Entered Searching State")
        self.manager.stop_current_process()
        # Launch or run the search node
        self.manager.run_node('my_package', 'search_for_object_node.py')
        # If we have a manual 30s approach, we can rely on that node's logic or timeouts.
        # Or rely on a /marker_detection_status callback to do FOUND_MARKER.

    def logVisualServoingStart(self, event=None):
        """
        On entering 'visual_servoing', stop the searching node
        and launch the pure PID node for visual servoing.
        """
        rospy.loginfo("[DroneStateMachine] Entered Visual Servoing State")
        self.manager.stop_current_process()
        # Run the new pure PID node
        self.manager.run_node('my_package', 'PID.py')

    def logTaskStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Performing Task State")
        self.manager.stop_current_process()
        # e.g. run a node that does the actual task
        self.manager.run_node('othmanPack', 'perform_task_node.py')

    # ---------------
    # MAVROS Helpers
    # ---------------
    def set_offboard_mode(self):
        """Sets the flight mode to OFFBOARD."""
        set_mode = SetModeRequest()
        set_mode.custom_mode = 'OFFBOARD'
        set_mode.base_mode = 0

        try:
            rospy.loginfo("[DroneStateMachine] Attempting OFFBOARD mode...")
            response = self.set_mode_client(set_mode)
            if response.mode_sent:
                rospy.loginfo("[DroneStateMachine] OFFBOARD mode set successfully.")
                return True
            else:
                rospy.logwarn("[DroneStateMachine] Failed to set OFFBOARD mode.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[DroneStateMachine] Error setting OFFBOARD mode: {e}")
            return False

    def arm_drone(self):
        """Arms the drone."""
        try:
            rospy.loginfo("[DroneStateMachine] Attempting to arm the drone.")
            response = self.arming_client(True)
            if response.success:
                rospy.loginfo("[DroneStateMachine] Drone armed successfully.")
                return True
            else:
                rospy.logwarn("[DroneStateMachine] Failed to arm the drone.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[DroneStateMachine] Service call failed while arming: {e}")
            return False

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("[DroneStateMachine] Shutting down node.")
        self.manager.stop_current_process()


if __name__ == '__main__':
    rospy.init_node('drone_state_machine_node', anonymous=True, log_level=rospy.INFO)
    goals = [
        (1, 2, 3),
        (0, -1, 2),
        (-4, -6, 4)
    ]
    sm = DroneStateMachine(goals)
    # Register the shutdown handler
    rospy.on_shutdown(sm.shutdown_handler)

    rospy.spin()
