#!/usr/bin/env python3
import rospy
import roslaunch
import subprocess
from transitions import Machine
from threading import Timer
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math
import signal
import sys


class StateManager:
    """
    This class is responsible for launching/stopping the actual ROS processes
    (like local_node.launch, PID node, search_for_object.py, etc.).
    Customize these as needed.
    """
    def __init__(self):
        self.current_process = None

    def launch_file(self, package, launch_file):
        """Launch a .launch file (e.g., local_node.launch)."""
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
        except Exception as e:
            rospy.logerr(f"[StateManager] Failed to launch {package}/{launch_file}: {e}")

    def run_node(self, package, executable):
        """Use rosrun to start a single node (Python script or otherwise)."""
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


class DroneStateMachine:
    states = [
        {'name': 'gps_navigation'},
        {'name': 'visual_servoing', 'on_enter': 'logVisualServoingStart'},
        {'name': 'performing_task', 'on_enter': 'logTaskStart'},
        {'name': 'search_for_object', 'on_enter': 'logSearchStart'}
    ]

    def __init__(self):
        # Manager that launches/stops processes
        self.manager = StateManager()

        # Positions
        self.current_position = None
        self.goal_position = None
        self.threshold = 0.2  # 0.2 meters to consider "reached goal"

        # Marker detection
        self.marker_detected = False

        # Build transitions
        self.machine = Machine(
            model=self,
            states=DroneStateMachine.states,
            initial='gps_navigation',
            auto_transitions=False,
            send_event=True
        )

        # Define transitions from gps_navigation
        self.machine.add_transition(
            trigger='REACHED_GOAL_POINT',
            source='gps_navigation',
            dest='visual_servoing',
            conditions='is_marker_detected'
        )
        self.machine.add_transition(
            trigger='REACHED_GOAL_POINT',
            source='gps_navigation',
            dest='search_for_object',
            unless='is_marker_detected'
        )

        # Transitions from visual_servoing
        self.machine.add_transition(
            trigger='OBJECT_NOT_FOUND',
            source='visual_servoing',
            dest='search_for_object'
        )
        self.machine.add_transition(
            trigger='REACHED_DESIRED_POSE',
            source='visual_servoing',
            dest='performing_task'
        )

        # Transition from performing_task
        self.machine.add_transition(
            trigger='TASK_DONE',
            source='performing_task',
            dest='gps_navigation'
        )

        # Transitions from search_for_object
        self.machine.add_transition(
            trigger='OBJECT_FOUND',
            source='search_for_object',
            dest='visual_servoing'
        )
        self.machine.add_transition(
            trigger='TIMEOUT',
            source='search_for_object',
            dest='gps_navigation'
        )

        # Timers
        self.search_timer = None
        self.object_not_found_timer = None
        self.object_found_timer = None

        # Start in GPS Navigation
        self.logGpsStart()

        # ROS Subscribers
        self._init_subscribers()

    def _init_subscribers(self):
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/marker_detection_status', Bool, self.marker_detection_callback)
        rospy.Subscriber('/visual_errors', Twist, self.visual_error_callback)
        rospy.loginfo("[DroneStateMachine] Subscribers initialized.")

    # --------------------
    # Callbacks
    # --------------------

    def marker_detection_callback(self, msg):
        """Update marker detection status."""
        self.marker_detected = msg.data

        # If in search_for_object and marker is detected, start 0.5s confirmation
        if self.state == 'search_for_object' and self.marker_detected:
            self.handle_object_found_event()

        # If in visual_servoing and we see the marker, reset lost-object timer
        if self.state == 'visual_servoing' and self.marker_detected:
            self.reset_object_not_found_timer()

    def is_marker_detected(self, event=None):
        return self.marker_detected

    def goal_callback(self, msg):
        """Update the goal_position from /goal_position (MarkerArray)."""
        if not msg.markers:
            rospy.logwarn("[DroneStateMachine] Received empty MarkerArray on /goal_position.")
            return

        marker = msg.markers[0]
        self.goal_position = marker.pose.position
        rospy.loginfo(f"[DroneStateMachine] Received new goal position: x={self.goal_position.x}, y={self.goal_position.y}, z={self.goal_position.z}")

    def position_callback(self, msg):
        """Monitor drone's position vs. goal in gps_navigation."""
        self.current_position = msg.pose.position
        if self.goal_position is None:
            return

        distance = self.calculate_distance(self.current_position, self.goal_position)
        if distance <= self.threshold and self.state == 'gps_navigation':
            rospy.loginfo("[DroneStateMachine] Drone reached the goal position within threshold.")
            self.goal_position = None
            rospy.loginfo("[DroneStateMachine] Triggering REACHED_GOAL_POINT")
            self.REACHED_GOAL_POINT()

    def visual_error_callback(self, msg):
        """
        If in visual_servoing, check if the error norm < 0.2 => REACHED_DESIRED_POSE.
        """
        if self.state == 'visual_servoing':
            x_err = msg.linear.x
            y_err = msg.linear.y
            z_err = msg.linear.z
            error_norm = math.sqrt(x_err**2 + y_err**2 + z_err**2)

            if error_norm < 0.2:
                rospy.loginfo("[DroneStateMachine] Visual errors < 0.2 -> REACHED_DESIRED_POSE()")
                self.REACHED_DESIRED_POSE()

    def calculate_distance(self, pos1, pos2):
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )

    # --------------------
    # State Entry Actions
    # --------------------

    def logVisualServoingStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Visual Servoing State")
        self.manager.stop_current_process()
        self.manager.run_node('othmanPack', 'PID.py')
        # Start the lost-object timer (5s), but reset whenever we see the marker
        self.reset_object_not_found_timer()

    def logTaskStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Performing Task State")
        self.manager.stop_current_process()
        self.manager.run_node('othmanPack', 'perform_task_node.py')

    def logSearchStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Search for Object State")
        self.manager.stop_current_process()
        self.manager.run_node('othmanPack', 'search_for_object_node.py')
        self.start_timeout_timer()

    def logGpsStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered GPS Navigation State")
        self.manager.stop_current_process()
        self.marker_detected = False
        # Launch local_node.launch (local planner, etc.)
        self.manager.launch_file('local_planner', 'local_node.launch')

    # --------------------
    # Timers for Visual Servoing and Searching
    # --------------------

    def reset_object_not_found_timer(self):
        """(Re)start a 5-second timer. If it expires => 5s of no marker => OBJECT_NOT_FOUND."""
        if self.object_not_found_timer:
            self.object_not_found_timer.cancel()

        self.object_not_found_timer = Timer(5.0, self._on_object_not_found)
        self.object_not_found_timer.start()
        rospy.loginfo("[DroneStateMachine] (Re)started 5-second lost-object timer.")

    def _on_object_not_found(self):
        """If we're still in visual_servoing, it means 5 continuous seconds with no marker."""
        if self.state == 'visual_servoing':
            rospy.logwarn("[DroneStateMachine] 5s with no marker detected -> OBJECT_NOT_FOUND.")
            self.OBJECT_NOT_FOUND()

    def start_timeout_timer(self):
        """30-second timer in search_for_object => TIMEOUT if we never confirm object found."""
        if self.search_timer:
            rospy.loginfo("[DroneStateMachine] Existing TIMEOUT timer canceled.")
            self.search_timer.cancel()

        self.search_timer = Timer(30.0, self._on_timeout)
        self.search_timer.start()
        rospy.loginfo("[DroneStateMachine] 30-second TIMEOUT timer started.")

    def _on_timeout(self):
        """Trigger TIMEOUT if still in search_for_object."""
        if self.state == 'search_for_object':
            rospy.logwarn("[DroneStateMachine] TIMEOUT timer expired -> TIMEOUT.")
            self.TIMEOUT()

    # --------------------
    # 0.5s Confirmation of Object Found in search_for_object
    # --------------------

    def handle_object_found_event(self):
        """If marker_detected is True in search_for_object, set a 0.5s confirmation window."""
        if self.object_found_timer and self.object_found_timer.is_alive():
            return
        self.object_found_timer = Timer(0.5, self._confirm_object_found)
        self.object_found_timer.start()
        rospy.loginfo("OBJECT_FOUND confirmation timer started.")

    def _confirm_object_found(self):
        """After 0.5s, if still in search_for_object and marker_detected = True => OBJECT_FOUND()."""
        if self.state == 'search_for_object' and self.marker_detected:
            rospy.loginfo("[DroneStateMachine] OBJECT_FOUND confirmed. Triggering OBJECT_FOUND.")
            self.OBJECT_FOUND()

    # --------------------
    # Shutdown Handling
    # --------------------

def shutdown_handler(signum, frame, sm):
    rospy.loginfo("Shutting down DroneStateMachine node.")
    sm.manager.stop_current_process()
    rospy.signal_shutdown("Manual shutdown")


if __name__ == '__main__':
    rospy.init_node('drone_state_machine_node', anonymous=True, log_level=rospy.INFO)
    sm = DroneStateMachine()
    signal.signal(signal.SIGINT, lambda s, f: shutdown_handler(s, f, sm))
    rospy.spin()
