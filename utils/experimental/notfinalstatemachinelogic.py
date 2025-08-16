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
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from tf.transformations import quaternion_from_euler



class StateManager:
    """
    This class is responsible for launching/stopping the actual ROS processes
    (like local_node.launch, combined_search_and_pid_node.py, performing_task_node.py, etc.).
    Customize these as needed.
    """
    def __init__(self,goal_points):
        self.current_process = None
        self.goal_points = goal_points
        self.goal_index = 0

    # def launch_file(self, package, launch_file):
    #     """Launch a .launch file (e.g., local_node.launch)."""
    #     rospy.loginfo(f"[StateManager] Attempting to launch file: {package}/{launch_file}")
    #     self.stop_current_process()
    #     try:
    #         uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    #         roslaunch.configure_logging(uuid)
    #         launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]
    #         self.current_process = roslaunch.parent.ROSLaunchParent(
    #             uuid, [launch_path],
    #             sigint_timeout=1.0, sigterm_timeout=1.0
    #         )
    #         self.current_process.start()
    #         rospy.loginfo(f"[StateManager] Launched: {package}/{launch_file}")
    #     except Exception as e:
    #         rospy.logerr(f"[StateManager] Failed to launch {package}/{launch_file}: {e}")


    def launch_file(self, package, launch_file, goal_x=None, goal_y=None, goal_z=None):
        """
        Launch a .launch file (e.g., local_node.launch) with optional goal coordinates.

        :param package: Name of the ROS package containing the launch file.
        :param launch_file: Name of the launch file to execute.
        :param goal_x: (Optional) X coordinate of the goal point.
        :param goal_y: (Optional) Y coordinate of the goal point.
        :param goal_z: (Optional) Z coordinate of the goal point.
        """
        rospy.loginfo(f"[StateManager] Attempting to launch file: {package}/{launch_file}")
        self.stop_current_process()
        try:
            # Generate a unique UUID for the roslaunch session
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)

            # Resolve the full path to the launch file
            launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]

            # Prepare launch arguments
            launch_args = []

            [goal_x, goal_y, goal_z] = [  self.goal_points[self.goal_index][0],  self.goal_points[self.goal_index][1], self.goal_points[self.goal_index][2] ]
            
            if self.goal_index < len(self.goal_points) -1:
                self.goal_index = self.goal_index + 1


            if goal_x is not None:
                launch_args.append(f"goal_x:={goal_x}")
            if goal_y is not None:
                launch_args.append(f"goal_y:={goal_y}")
            if goal_z is not None:
                launch_args.append(f"goal_z:={goal_z}")

            # Combine the launch file path with its arguments
            launch_files = [launch_path] + launch_args

            # Initialize the ROSLaunchParent with the launch files and arguments
            self.current_process = roslaunch.parent.ROSLaunchParent(
                uuid, launch_files,
                sigint_timeout=1.0, sigterm_timeout=1.0
            )

            # Start the launch process
            self.current_process.start()
            rospy.loginfo(f"[StateManager] Launched: {package}/{launch_file} with args goal_x={goal_x}, goal_y={goal_y}, goal_z={goal_z}")

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
                    #self.current_process.join()
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
        {'name': 'gps_navigation', 'on_enter': 'logGpsStart'},
        {'name': 'visual_servoing', 'on_enter': 'logVisualServoingStart'},
        {'name': 'performing_task', 'on_enter': 'logTaskStart'},
    ]

    def __init__(self, goal_points):
        # Manager that launches/stops processes
        self.manager = StateManager(goal_points)

        # Positions
        self.current_position = None
        self.goal_position = None
        self.threshold = 0.2  # 0.2 meters to consider "reached goal"

        # Marker detection
        self.marker_detected = False

        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        rospy.loginfo("[DroneStateMachine] Service proxies for set_mode and arming initialized.")

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

        # Define timeout transition within visual_servoing
        self.machine.add_transition(
            trigger='TIMEOUT',
            source='visual_servoing',
            dest='gps_navigation'
        )

        # Timers
        # self.object_not_found_timer = None

        # Start in GPS Navigation
        self.logGpsStart()

        # ROS Subscribers
        self._init_subscribers()

    def _init_subscribers(self):
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/marker_detection_status', Bool, self.marker_detection_callback)
        rospy.Subscriber('/visual_errors', Twist, self.visual_error_callback)
        rospy.Subscriber('/search_timeout', Bool, self.search_timeout_callback)  # Added Subscriber
        rospy.Subscriber('/task_done', Bool, self.task_done_callback)  # Added Subscriber
        rospy.loginfo("[DroneStateMachine] Subscribers initialized.")

    # --------------------
    # Callbacks
    # --------------------

    def marker_detection_callback(self, msg):
        """Update marker detection status."""
        self.marker_detected = msg.data

        # If in visual_servoing and marker is detected, reset lost-object timer
        # if self.state == 'visual_servoing':
        #     if self.marker_detected:
        #         self.reset_object_not_found_timer()
        #     else:
        #         self.start_object_not_found_timer()

    def search_timeout_callback(self, msg):
        """Handle search timeout by triggering TIMEOUT if True and in visual_servoing state."""
        if msg.data and self.state == 'visual_servoing':
            rospy.loginfo("[DroneStateMachine] Received /search_timeout=True, triggering TIMEOUT.")
            self.TIMEOUT()
        elif msg.data:
            rospy.logwarn("[DroneStateMachine] Received /search_timeout=True, but not in 'visual_servoing' state.")


    def task_done_callback(self, msg):
        """Handle task completion by triggering TASK_DONE if True and in performing_task state."""
        if msg.data and self.state == 'performing_task':
            rospy.loginfo("[DroneStateMachine] Received /task_done=True, triggering TASK_DONE.")
            self.TASK_DONE()
        elif msg.data:
            rospy.logwarn("[DroneStateMachine] Received /task_done=True, but not in 'performing_task' state.")


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

            # Corrected condition
            if abs(1.2 - error_norm) < 0.2:
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
        self.manager.run_node('othmanPack', 'finalPID.py')  # Assuming 'finalPID.py' is your combined node
        # Start the lost-object timer (5s), but reset whenever we see the marker
        # self.reset_object_not_found_timer()

    def logTaskStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Performing Task State")
        self.manager.stop_current_process()
        self.manager.run_node('othmanPack', 'perform_task_node.py')

    def logGpsStart(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered GPS Navigation State")
        self.manager.stop_current_process()
        self.marker_detected = False
        # Launch local_node.launch (local planner, etc.)
        self.manager.launch_file('local_planner', 'local_node.launch')
        rospy.loginfo("[DroneStateMachine] GPS Navigation Node Launched")
        
        # Wait to ensure setpoints are being published
        rospy.sleep(2)  # Adjust the sleep duration as needed based on your setpoint publisher's startup time
        
        # Attempt to set mode to OFFBOARD
        if self.set_offboard_mode():
            rospy.sleep(1)  # Short delay before arming
            if self.arm_drone():
                rospy.loginfo("[DroneStateMachine] Drone armed successfully and in OFFBOARD mode.")
            else:
                rospy.logwarn("[DroneStateMachine] Drone failed to arm.")
        else:
            rospy.logwarn("[DroneStateMachine] OFFBOARD mode not set. Retrying...")
            # Optional: Implement retry logic here

    def set_offboard_mode(self):
        """Sets the flight mode to OFFBOARD."""
        mode = 'OFFBOARD'
        set_mode = SetModeRequest()
        set_mode.custom_mode = mode
        set_mode.base_mode = 0  # Ensure base_mode is set to 0
        
        try:
            rospy.loginfo(f"[DroneStateMachine] Attempting to set mode to {mode}")
            response = self.set_mode_client(set_mode)
            if response.mode_sent:
                rospy.loginfo(f"[DroneStateMachine] OFFBOARD mode set successfully.")
                return True
            else:
                rospy.logwarn(f"[DroneStateMachine] Failed to set OFFBOARD mode.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[DroneStateMachine] Service call failed while setting mode: {e}")
            return False

        
    def arm_drone(self):
        """Arms the drone."""
        try:
            rospy.loginfo("[DroneStateMachine] Attempting to arm the drone.")
            response = self.arming_client(True)  # True to arm, False to disarm
            if response.success:
                rospy.loginfo("[DroneStateMachine] Drone armed successfully.")
                return True
            else:
                rospy.logwarn("[DroneStateMachine] Failed to arm the drone.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[DroneStateMachine] Service call failed while arming: {e}")
            return False



    # --------------------
    # Timers for Visual Servoing
    # --------------------

    # def reset_object_not_found_timer(self):
    #     """Cancel and reset the object_not_found_timer."""
    #     if self.object_not_found_timer:
    #         self.object_not_found_timer.cancel()

    #     self.object_not_found_timer = Timer(5.0, self._on_object_not_found)
    #     self.object_not_found_timer.start()
    #     rospy.loginfo("[DroneStateMachine] (Re)started 5-second lost-object timer.")

    # def start_object_not_found_timer(self):
    #     """Start the object_not_found_timer if it's not already running."""
    #     if not self.object_not_found_timer or not self.object_not_found_timer.is_alive():
    #         self.object_not_found_timer = Timer(5.0, self._on_object_not_found)
    #         self.object_not_found_timer.start()
    #         rospy.loginfo("[DroneStateMachine] Started 5-second lost-object timer.")

    # def _on_object_not_found(self):
    #     """If we're still in visual_servoing, it means 5 continuous seconds with no marker."""
    #     if self.state == 'visual_servoing':
    #         rospy.logwarn("[DroneStateMachine] 5s with no marker detected -> TIMEOUT.")
    #         self.TIMEOUT()

    # --------------------
    # Shutdown Handling
    # --------------------

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("Shutting down DroneStateMachine node.")
        self.manager.stop_current_process()
        rospy.signal_shutdown("Manual shutdown")


def shutdown_handler(signum, frame, sm):
    rospy.loginfo("Shutting down DroneStateMachine node.")
    sm.manager.stop_current_process()
    rospy.signal_shutdown("Manual shutdown")


if __name__ == '__main__':
    goals = [
        (1,2,3),
        (-6, -5, 4),
        (0,0,2)
    ]
    rospy.init_node('drone_state_machine_node', anonymous=True, log_level=rospy.INFO)
    sm = DroneStateMachine(goals)
    signal.signal(signal.SIGINT, lambda s, f: sm.shutdown_handler(s, f))
    rospy.spin()
