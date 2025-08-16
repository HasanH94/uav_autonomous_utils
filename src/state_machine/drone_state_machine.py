#!/usr/bin/env python3
import rospy
import roslaunch
import subprocess
import os
from transitions import Machine
from threading import Timer, Lock
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Header
import math
import signal
import sys
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.msg import State # Added import
from tf.transformations import quaternion_from_euler


class StateManager:
    """
    This class is responsible for launching/stopping the actual ROS processes
    (like local_node.launch, combined_search_and_pid_node.py, performing_task_node.py, etc.).
    """
    def __init__(self):
        self.current_process = None
        self.process_lock = Lock()

    def launch_file(self, package, launch_file, publish_topic=None, *args):
        """Launch a .launch file."""
        rospy.loginfo(f"[StateManager] Attempting to launch file: {package}/{launch_file}")
        self.stop_current_process()
        rospy.sleep(2.0) # Give time for previous processes to terminate
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

            if len(args) == 3 and all(isinstance(arg, (int, float)) for arg in args):
                self.publish_goal(*args, publish_topic)
            elif args:
                rospy.logwarn("[StateManager] Invalid arguments for publish_goal. Expected three numerical values.")
            rospy.sleep(1)
        except Exception as e:
            rospy.logerr(f"[StateManager] Failed to launch {package}/{launch_file}: {e}")

    def run_node(self, package, executable):
        """Use rosrun to start a single node (Python script or otherwise)."""
        rospy.loginfo(f"[StateManager] Attempting to run node: {package}/{executable}")
        self.stop_current_process()
        try:
            command = f"rosrun {package} {executable}"
            self.current_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid) # Use setsid for process group
            rospy.loginfo(f"[StateManager] Started node via: {command}")
        except Exception as e:
            rospy.logerr(f"[StateManager] Failed to run node {package}/{executable}: {e}")

    def stop_current_process(self):
        """Stop any currently running process or launch file."""
        with self.process_lock:
            if self.current_process:
                rospy.loginfo(f"[StateManager] Stopping current process: {self.current_process}")
                try:
                    if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
                        self.current_process.shutdown()
                        rospy.loginfo("[StateManager] roslaunch parent shutdown complete.")
                        rospy.sleep(2.0) # Add a small delay for graceful shutdown
                    else:
                        # For subprocess.Popen, send SIGINT to the process group
                        os.killpg(os.getpgid(self.current_process.pid), signal.SIGINT)
                        # Wait for process to terminate with a loop
                        timeout_start = rospy.Time.now()
                        while (rospy.Time.now() - timeout_start).to_sec() < 10.0: # 10 seconds total timeout
                            if self.current_process.poll() is not None: # Check if process has terminated
                                rospy.loginfo("[StateManager] Subprocess terminated gracefully.")
                                break
                            rospy.sleep(0.1) # Check every 100ms
                        else:
                            rospy.logwarn("[StateManager] Subprocess did not terminate within 10 seconds. Forcing kill.")
                            self.current_process.kill() # Force kill if not terminated
                        rospy.loginfo("[StateManager] Terminated subprocess.")
                    rospy.loginfo("[StateManager] Stopped current process.")
                except (ProcessLookupError, subprocess.TimeoutExpired) as e:
                    rospy.logwarn(f"[StateManager] Process did not terminate gracefully ({e}). Forcing kill.")
                    self.current_process.kill() # Force kill if not terminated
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

        rate = rospy.Rate(10)
        timeout_seconds = 5
        start_time = rospy.Time.now()
        while goal_pub.get_num_connections() < 1 and not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout_seconds:
                rospy.logwarn("No subscribers connected after 5s, proceeding anyway.")
                break
            rate.sleep()

        goal = PoseStamped()
        goal.header = Header()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom" # Changed from "map" to "odom"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.w = 1.0 # No rotation

        goal_pub.publish(goal)
        rospy.loginfo(f"[StateManager] Published goal to {publish_topic}: x={x}, y={y}, z={z}")


class DroneStateMachine(Machine):
    states = [
        {'name': 'gps_navigation', 'on_enter': 'on_enter_gps_navigation'},
        {'name': 'search_for_object', 'on_enter': 'on_enter_search_for_object'},
        {'name': 'visual_servoing', 'on_enter': 'on_enter_visual_servoing'},
        {'name': 'performing_task', 'on_enter': 'on_enter_performing_task'},
        {'name': 'returning_home', 'on_enter': 'on_enter_returning_home'},
        {'name': 'landing', 'on_enter': 'on_enter_landing'}, # New state for mission completion
    ]

    def __init__(self):
        super().__init__(
            model=self,
            states=DroneStateMachine.states,
            initial='gps_navigation',
            auto_transitions=False,
            send_event=True
        )

        self.manager = StateManager()
        self.goal_points = rospy.get_param('~goal_points', [])
        self.home_position = rospy.get_param('~home_position', [0.0, 0.0, 2.0])
        self.gps_grace_period = rospy.get_param('~gps_grace_period', 5.0)

        self.current_mission_goal_index = 0

        self.current_drone_pose = None
        self.aruco_is_detected = False
        self.mavros_current_mode = ""
        self.aruco_lost_timer = None
        self.gps_navigation_active = False # New flag to indicate if GPS navigation is fully active
        self.moving_to_next_gps_goal = False # New flag to indicate if drone is moving to next GPS goal

        # New robustness features
        self.aruco_confidence = 0
        self.ARUCO_CONFIDENCE_THRESHOLD = rospy.get_param('~aruco_confidence_threshold', 20)  # e.g., 2 seconds at 10Hz
        self.aruco_lost_delay = rospy.get_param('~aruco_lost_delay', 2.0) # 2-second delay
        self.aruco_pose = None
        self.MAX_VISUAL_SERVO_DISTANCE = rospy.get_param('~max_visual_servo_distance', 5.0) # 5 meters

        # ROS Publishers
        self.mission_goal_pub = rospy.Publisher('/move_base_mission', PoseStamped, queue_size=10)
        self.visual_goal_pub = rospy.Publisher('/move_base_visual', PoseStamped, queue_size=10)

        # ROS Services
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        rospy.loginfo("[DroneStateMachine] Service proxies for set_mode and arming initialized.")

        # --- Transitions ---
        # GPS Navigation related
        self.add_transition(
            trigger='gps_goal_reached',
            source='gps_navigation',
            dest='visual_servoing',
            conditions=['is_aruco_detected'],
            before='increment_goal_index',
            after=['publish_visual_goal', 'on_exit_gps_navigation']
        )
        self.add_transition(
            trigger='gps_goal_reached',
            source='gps_navigation',
            dest='search_for_object',
            unless=['is_aruco_detected'],
            before='increment_goal_index',
            after='on_exit_gps_navigation'
        )
        self.add_transition(
            trigger='mission_complete',
            source='gps_navigation',
            dest='landing',
            conditions=['is_mission_complete'],
            after='on_exit_gps_navigation'
        )

        # Visual Servoing related
        self.add_transition(
            trigger='aruco_found_mid_flight',
            source=['gps_navigation', 'search_for_object'],
            dest='visual_servoing',
            conditions=['is_aruco_detected', 'is_aruco_within_range'],
            unless=['is_moving_to_next_gps_goal'],
            after='publish_visual_goal'
        )
        self.add_transition(
            trigger='visual_servoing_goal_reached',
            source='visual_servoing',
            dest='performing_task'
        )
        self.add_transition(
            trigger='aruco_lost',
            source='visual_servoing',
            dest='search_for_object',
            unless=['is_aruco_detected'] # If marker lost, go search
        )

        # Search for Object related
        self.add_transition(
            trigger='search_timeout',
            source='search_for_object',
            dest='gps_navigation', # Go back to GPS if search times out
            before='decrement_goal_index_if_needed' # Re-attempt previous GPS goal
        )

        # Performing Task related
        self.add_transition(
            trigger='task_done',
            source='performing_task',
            dest='gps_navigation',
            unless=['is_mission_finished'],
            before='increment_goal_index',
            after='set_moving_to_next_gps_goal_true'
        )
        self.add_transition(
            trigger='task_done',
            source='performing_task',
            dest='returning_home',
            conditions=['is_mission_finished']
        )

        # Return to Home related
        self.add_transition(
            trigger='home_reached',
            source='returning_home',
            dest='landing'
        )

        # --- ROS Subscribers ---
        self._init_subscribers()

        rospy.loginfo(f"Drone State Machine initialized. Initial state: {self.state}")
        self.on_enter_gps_navigation()

    def _init_subscribers(self):
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/aruco_detection_status', Bool, self.aruco_detection_status_callback)
        rospy.Subscriber('/move_base_visual', PoseStamped, self.aruco_pose_callback)
        rospy.Subscriber('/drone_events/reached_desired_pose', Bool, self.visual_servoing_goal_reached_callback)
        rospy.Subscriber('/drone_events/reached_goal_point', Bool, self.gps_goal_reached_callback)
        rospy.Subscriber('/drone_events/task_done', Bool, self.task_done_callback)
        rospy.Subscriber('/drone_events/search_timeout', Bool, self.search_timeout_callback) # Assuming a search node publishes this

        rospy.loginfo("[DroneStateMachine] Subscribers initialized.")

    # --- MAVROS Callbacks ---
    def mavros_state_callback(self, msg):
        self.mavros_current_mode = msg.mode

    def position_callback(self, msg):
        self.current_drone_pose = msg

    def on_enter_returning_home(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Returning Home State")
        self.manager.launch_file('othmanPack', 'gps_navigation.launch')
        self.arm_drone()
        self.publish_mission_goal(self.home_position[0], self.home_position[1], self.home_position[2])

    def aruco_pose_callback(self, msg):
        self.aruco_pose = msg.pose

    def aruco_detection_status_callback(self, msg):
        # Log the raw message data and current state for debugging
        rospy.loginfo(f"[DEBUG] aruco_detection_status_callback: msg.data={msg.data}, current_state={self.state}")
        
        if msg.data:
            self.aruco_confidence = min(self.ARUCO_CONFIDENCE_THRESHOLD, self.aruco_confidence + 2) # Increase faster than it decreases
        else:
            self.aruco_confidence = max(0, self.aruco_confidence - 1) # Decrease on missed frames

        self.aruco_is_detected = self.aruco_confidence >= self.ARUCO_CONFIDENCE_THRESHOLD

        # Cancel the timer if the marker is found again
        if self.aruco_is_detected and self.aruco_lost_timer is not None:
            self.aruco_lost_timer.cancel()
            self.aruco_lost_timer = None
            rospy.loginfo("[DEBUG] Aruco marker re-detected, search timer cancelled.")
        
        # Trigger immediate transition if marker found mid-flight
        if self.aruco_is_detected and (self.state == 'search_for_object' or (self.gps_navigation_active and self.state == 'gps_navigation')):
            rospy.loginfo(f"[DEBUG] Aruco detected ({self.aruco_is_detected}) in {self.state}. Triggering aruco_found_mid_flight.")
            self.aruco_found_mid_flight()
        elif not self.aruco_is_detected and self.state == 'visual_servoing':
            rospy.loginfo(f"[DEBUG] Aruco lost ({self.aruco_is_detected}) in {self.state}. Starting timer before transitioning to search.")
            if self.aruco_lost_timer is None:
                self.aruco_lost_timer = Timer(self.aruco_lost_delay, self.aruco_lost)
                self.aruco_lost_timer.start()


    # --- Event Callbacks (Triggers) ---
    def visual_servoing_goal_reached_callback(self, msg):
        rospy.loginfo(f"[DEBUG] visual_servoing_goal_reached_callback: msg.data={msg.data}, current_state={self.state}")
        if msg.data and self.state == 'visual_servoing':
            rospy.loginfo("Event: VISUAL_SERVOING_GOAL_REACHED received.")
            self.visual_servoing_goal_reached()

    def gps_goal_reached_callback(self, msg):
        rospy.loginfo(f"[DEBUG] gps_goal_reached_callback: msg.data={msg.data}, current_state={self.state}")
        if msg.data:
            if self.state == 'gps_navigation':
                rospy.loginfo("Event: GPS_GOAL_REACHED received.")
                self.gps_goal_reached()
            elif self.state == 'returning_home':
                rospy.loginfo("Event: HOME_REACHED received.")
                self.home_reached()

    def task_done_callback(self, msg):
        rospy.loginfo(f"[DEBUG] task_done_callback: msg.data={msg.data}, current_state={self.state}")
        if msg.data and self.state == 'performing_task':
            rospy.loginfo("Event: TASK_DONE received.")
            self.task_done()

    def search_timeout_callback(self, msg):
        rospy.loginfo(f"[DEBUG] search_timeout_callback: msg.data={msg.data}, current_state={self.state}")
        if msg.data and self.state == 'search_for_object':
            rospy.loginfo("Event: SEARCH_TIMEOUT received.")
            self.search_timeout()

    # --- Conditionals for Transitions ---
    def is_aruco_detected(self, event=None):
        rospy.loginfo(f"[DEBUG] Condition: is_aruco_detected -> {self.aruco_is_detected}")
        return self.aruco_is_detected

    def is_aruco_within_range(self, event=None):
        if self.aruco_pose is None or self.current_mission_goal_index >= len(self.goal_points):
            return False

        current_goal_coords = self.goal_points[self.current_mission_goal_index]
        current_goal_pose = PoseStamped()
        current_goal_pose.pose.position.x = current_goal_coords[0]
        current_goal_pose.pose.position.y = current_goal_coords[1]
        current_goal_pose.pose.position.z = current_goal_coords[2]

        distance = self.calculate_distance(self.aruco_pose.position, current_goal_pose.pose.position)
        rospy.loginfo(f"[DEBUG] Distance to ArUco marker: {distance:.2f}m")
        return distance <= self.MAX_VISUAL_SERVO_DISTANCE

    def is_mission_finished(self, event=None):
        is_finished = self.current_mission_goal_index + 1 >= len(self.goal_points)
        rospy.loginfo(f"[DEBUG] Condition: is_mission_finished -> {is_finished}")
        return is_finished

    def is_moving_to_next_gps_goal(self, event=None):
        rospy.loginfo(f"[DEBUG] Condition: is_moving_to_next_gps_goal -> {self.moving_to_next_gps_goal}")
        return self.moving_to_next_gps_goal

    # --- Helper Functions ---
    def calculate_distance(self, pos1, pos2):
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )

    def set_mavros_mode(self, mode):
        rospy.loginfo(f"Attempting to set MAVROS mode to {mode}.")
        rate = rospy.Rate(10) # 10 Hz
        timeout_t = rospy.Time.now() + rospy.Duration(5) # 5 second timeout

        while not rospy.is_shutdown() and self.mavros_current_mode != mode and rospy.Time.now() < timeout_t:
            try:
                set_mode_req = SetModeRequest()
                set_mode_req.custom_mode = mode
                resp = self.set_mode_client.call(set_mode_req)
                if resp.mode_sent:
                    rospy.loginfo(f"MAVROS mode change request for {mode} sent.")
                else:
                    rospy.logwarn(f"Failed to send MAVROS mode change request for {mode}. Mode_sent: {resp.mode_sent}")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call to set_mode failed: {e}")

            rate.sleep()

        if self.mavros_current_mode == mode:
            rospy.loginfo(f"Successfully set to {mode} mode.")
            return True
        else:
            rospy.logwarn(f"Failed to set {mode} mode after timeout. Current mode: {self.mavros_current_mode}")
            return False

    def arm_drone(self):
        try:
            rospy.loginfo("Attempting to arm the drone.")
            response = self.arming_client(True)
            if response.success:
                rospy.loginfo("Drone armed successfully.")
                return True
            else:
                rospy.logwarn("Failed to arm the drone.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed while arming: {e}")
            return False

    def _check_gps_goal_reached(self):
        if self.current_drone_pose is None or self.current_mission_goal_index >= len(self.goal_points):
            return

        current_goal_coords = self.goal_points[self.current_mission_goal_index]
        current_goal_pose = PoseStamped()
        current_goal_pose.pose.position.x = current_goal_coords[0]
        current_goal_pose.pose.position.y = current_goal_coords[1]
        current_goal_pose.pose.position.z = current_goal_coords[2]

        distance = self.calculate_distance(self.current_drone_pose.pose.position, current_goal_pose.pose.position)

        rospy.loginfo(f"[DEBUG] GPS Goal Check: Dist={distance:.2f}m, Threshold={self.goal_reached_threshold:.2f}m, ArucoDetected={self.aruco_is_detected}")

        if distance <= self.goal_reached_threshold:
            rospy.loginfo(f"Drone reached mission goal {self.current_mission_goal_index + 1}. Distance: {distance:.2f}m")
            # Trigger the GPS goal reached event
            if self.is_mission_complete():
                self.mission_complete()
            else:
                self.gps_goal_reached() # This trigger will check is_aruco_detected condition

    # --- Actions (on_enter, before, after) ---
    def increment_goal_index(self, event=None):
        self.current_mission_goal_index += 1
        rospy.loginfo(f"Incremented mission goal index to {self.current_mission_goal_index}")

    def decrement_goal_index_if_needed(self):
        # If search times out, we might want to re-attempt the previous GPS goal
        # Or, if it's the first goal, stay at the first goal.
        if self.current_mission_goal_index > 0:
            self.current_mission_goal_index -= 1
            rospy.loginfo(f"Search timed out, re-attempting previous GPS goal. Index: {self.current_mission_goal_index}")
        else:
            rospy.loginfo("Search timed out for first goal, staying at first goal.")

    def publish_mission_goal(self, x=None, y=None, z=None):
        if x is not None and y is not None and z is not None:
            goal_coords = (x, y, z)
        elif self.current_mission_goal_index < len(self.goal_points):
            goal_coords = self.goal_points[self.current_mission_goal_index]
        else:
            rospy.logwarn("Attempted to publish mission goal when all mission goals are completed.")
            return

        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.pose.position.x = goal_coords[0]
        goal.pose.position.y = goal_coords[1]
        goal.pose.position.z = goal_coords[2]
        goal.pose.orientation.w = 1.0 # No rotation

        rate = rospy.Rate(1)
        timeout_t = rospy.Time.now() + rospy.Duration(10)
        while self.mission_goal_pub.get_num_connections() == 0 and rospy.Time.now() < timeout_t:
            rospy.loginfo("Waiting for /move_base_mission subscriber...")
            rate.sleep()

        if self.mission_goal_pub.get_num_connections() == 0:
            rospy.logwarn("No subscriber to /move_base_mission after timeout. Goal may not be received.")
        else:
            rospy.loginfo("Subscriber to /move_base_mission detected. Publishing goal.")

        rospy.loginfo(f"Publishing mission goal: {goal_coords}. Timestamp: {rospy.Time.now()}")
        self.mission_goal_pub.publish(goal)
        rospy.loginfo(f"Goal published to /move_base_mission: {goal.pose.position.x}, {goal.pose.position.y}, {goal.pose.position.z}")

    def publish_visual_goal(self, event=None):
        if self.current_mission_goal_index < len(self.goal_points):
            goal_coords = self.goal_points[self.current_mission_goal_index]
            goal = PoseStamped()
            goal.header.frame_id = "odom"
            goal.pose.position.x = goal_coords[0]
            goal.pose.position.y = goal_coords[1]
            goal.pose.position.z = goal_coords[2]
            goal.pose.orientation.w = 1.0 # No rotation

            rate = rospy.Rate(1)
            timeout_t = rospy.Time.now() + rospy.Duration(5)
            while self.visual_goal_pub.get_num_connections() == 0 and rospy.Time.now() < timeout_t:
                rospy.loginfo("Waiting for /move_base_visual subscriber...")
                rate.sleep()

            if self.visual_goal_pub.get_num_connections() == 0:
                rospy.logwarn("No subscriber to /move_base_visual after timeout. Visual goal may not be received.")
            else:
                rospy.loginfo("Subscriber to /move_base_visual detected. Publishing visual goal.")

            self.visual_goal_pub.publish(goal)
            rospy.loginfo(f"Published current mission goal to /move_base_visual: {goal_coords}")
        else:
            rospy.logwarn("Attempted to publish visual goal when all mission goals are completed.")

    

    # --- on_enter methods for states ---
    def on_enter_gps_navigation(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered GPS Navigation State")
        self.set_moving_to_next_gps_goal_true() # Ensure grace period is always active on entry
        self.manager.launch_file('othmanPack', 'gps_navigation.launch')
        self.arm_drone()
        self.publish_mission_goal() # Publish the current mission goal
        self.gps_navigation_active = True
        # Reset moving_to_next_gps_goal after a short delay to allow initial movement
        rospy.Timer(rospy.Duration(self.gps_grace_period), self.reset_moving_to_next_gps_goal, oneshot=True)

    def on_enter_search_for_object(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Search for Object State")
        self.manager.launch_file('othmanPack', 'search_for_object.launch') # Assuming this is a launch file
        self.arm_drone()

    def on_enter_visual_servoing(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Visual Servoing State")
        self.manager.launch_file('othmanPack', 'visual_survoing_with_avoidance.launch') # Assuming this is a launch file
        self.arm_drone()
        # publish_visual_goal is called via 'after' callback in transition

    def on_enter_performing_task(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Performing Task State")
        self.manager.launch_file('othmanPack', 'performing_task.launch') # Assuming this is a launch file
        self.arm_drone()

    def on_enter_landing(self, event=None):
        rospy.loginfo("[DroneStateMachine] Entered Landing State. Mission Complete.")
        self.manager.stop_current_process() # Stop any active controller
        if self.set_mavros_mode('AUTO.LAND'):
            rospy.loginfo("Landing mode set. Waiting 10 seconds for landing to initiate before shutdown.")
            rospy.sleep(10.0)
        else:
            rospy.logwarn("Failed to set landing mode. Shutting down anyway.")
        rospy.signal_shutdown("Mission completed and drone is landing.")

    def on_exit_gps_navigation(self, event=None):
        rospy.loginfo("[DroneStateMachine] Exited GPS Navigation State.")
        self.gps_navigation_active = False

    def set_moving_to_next_gps_goal_true(self, event=None):
        rospy.loginfo("Setting moving_to_next_gps_goal to True.")
        self.moving_to_next_gps_goal = True

    def reset_moving_to_next_gps_goal(self, event=None):
        rospy.loginfo("Resetting moving_to_next_gps_goal to False.")
        self.moving_to_next_gps_goal = False

    # --- Shutdown Handling ---
    def shutdown_handler(self, *args):
        rospy.loginfo("Ctrl+C received. Shutting down DroneStateMachine node.")
        self.manager.stop_current_process()
        # Set mode to AUTO.LOITER before final shutdown
        self.set_mavros_mode('AUTO.LOITER')
        rospy.signal_shutdown("Manual shutdown via Ctrl+C")


if __name__ == '__main__':
    rospy.init_node('drone_state_machine_node', anonymous=True, log_level=rospy.INFO)
    sm = DroneStateMachine()
    
    # Register the shutdown handler
    rospy.on_shutdown(sm.shutdown_handler) # Also register with rospy for other shutdown events
    
    rospy.spin()