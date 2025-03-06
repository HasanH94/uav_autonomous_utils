#!/usr/bin/env python3
from transitions import Machine
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
import roslaunch
import subprocess
import threading

class StateManager:
    def __init__(self):
        # IMPORTANT: Node init in the main thread
        rospy.init_node('state_manager_node', anonymous=True)
        self.current_process = None

    def launch_file(self, package, launch_file):
        """
        Launch a .launch file safely from the main thread. 
        """
        if self.current_process:
            self.stop_current_process()

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]

        # Disabling signal handling so we don't trigger the "signal only works in main thread" in certain contexts:
        self.current_process = roslaunch.parent.ROSLaunchParent(
            uuid, [launch_path],
            sigint_timeout=0,
            sigterm_timeout=0
        )

        self.current_process.start()
        rospy.loginfo(f"Started launch file: {package}/{launch_file}")

    def run_node(self, package, executable, node_name=None):
        """
        Run a single node with rosrun. 
        """
        if self.current_process:
            self.stop_current_process()

        command = f"rosrun {package} {executable}"
        self.current_process = subprocess.Popen(command, shell=True)
        rospy.loginfo(f"Started node: {package}/{executable}")

    def stop_current_process(self):
        if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
            self.current_process.shutdown()
        elif isinstance(self.current_process, subprocess.Popen):
            self.current_process.terminate()
            self.current_process.wait()

        self.current_process = None
        rospy.loginfo("Stopped current process")

    def transition_to_state(self, state_name):
        """
        Start or stop processes depending on the state.
        """
        if state_name == 'visual_servoing':
            self.run_node('othmanPack', 'PID.py', 'visual_servoing_pid_node')

        elif state_name == 'gps_navigation':
            # Launch local_node.launch which sets up the local planner in Gazebo
            # Here we disable signal handling for safety.
            self.launch_file('local_planner', 'local_node.launch')

        elif state_name == 'performing_task':
            self.run_node('othmanPack', 'perform_task_node.py', 'perform_task_node')

        elif state_name == 'search_for_object':
            self.run_node('othmanPack', 'search_for_object_node.py', 'search_for_object_offboard_node')

        else:
            rospy.logwarn(f"Unknown state: {state_name}")


class DroneStateMachine:
    states = [
        'gps_navigation',
        'visual_servoing',
        'performing_task',
        'search_for_object'
    ]

    def __init__(self):
        self.state_manager = StateManager()

        # Initialize transitions state machine
        self.machine = Machine(
            model=self, 
            states=DroneStateMachine.states, 
            initial='gps_navigation'
        )

        # Add transitions
        self.machine.add_transition(
            trigger='object_detected',
            source='gps_navigation',
            dest='visual_servoing',
            after='start_visual_servoing'
        )
        self.machine.add_transition(
            trigger='reached_desired_pose',
            source='visual_servoing',
            dest='performing_task',
            after='perform_task'
        )
        self.machine.add_transition(
            trigger='object_lost',
            source='visual_servoing',
            dest='search_for_object',
            after='start_search_with_timer'
        )
        self.machine.add_transition(
            trigger='object_found',
            source='search_for_object',
            dest='visual_servoing',
            after='start_visual_servoing'
        )
        self.machine.add_transition(
            trigger='task_done',
            source='performing_task',
            dest='gps_navigation',
            after='start_gps'
        )
        self.machine.add_transition(
            trigger='timeout',
            source='search_for_object',
            dest='gps_navigation',
            after='start_gps'
        )

        # Subscribers
        rospy.Subscriber('/marker_detection_status', Bool, self.detection_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        rospy.Subscriber('/task_done', Bool, self.task_done_callback)

        # Pose tracking
        self.desired_pose = PoseStamped()
        self.desired_pose_threshold = 0.1

        # Timers / flags
        self.search_timer = None

        # Hysteresis for object lost
        self.object_lost_time = None
        self.lost_duration_threshold = 5.0

        # 3-second delay is stored; when transitions are triggered, we set a "pending transition"
        self.pending_transition = None
        self.pending_transition_time = None

        # For the Timer-based transitions (e.g. "timeout"), 
        # we store a simple boolean that the main loop checks.
        self.search_timeout_flag = False

    # -----------------------------
    # Callbacks from ROS topics
    # -----------------------------

    def detection_callback(self, msg):
        if msg.data:
            # Object detected
            self.object_lost_time = None  # reset lost timer
            if self.state == 'gps_navigation':
                self.object_detected()
                rospy.loginfo("Switching to Visual Servoing from GPS Navigation.")
            elif self.state == 'search_for_object':
                self.object_found()
                rospy.loginfo("Object found, switching to Visual Servoing.")
                if self.search_timer:
                    self.search_timer.shutdown()
        else:
            # Object not detected
            if self.state == 'visual_servoing':
                if self.object_lost_time is None:
                    self.object_lost_time = rospy.Time.now()
                else:
                    lost_time_elapsed = (rospy.Time.now() - self.object_lost_time).to_sec()
                    if lost_time_elapsed >= self.lost_duration_threshold:
                        # Actually lost for > 5s
                        self.object_lost()
                        rospy.loginfo("Lost object for >5s, switching to Search for Object.")
                        self.object_lost_time = None

    def pose_callback(self, pose_msg):
        if self.state == 'visual_servoing' and self.is_at_desired_pose(pose_msg):
            self.reached_desired_pose()
            rospy.loginfo("Reached Desired Pose, Performing Task")

    def goal_callback(self, msg):
        if msg.markers:
            goal_marker = msg.markers[0]
            self.desired_pose.pose.position.x = goal_marker.pose.position.x
            self.desired_pose.pose.position.y = goal_marker.pose.position.y
            self.desired_pose.pose.position.z = goal_marker.pose.position.z
            orientation_q = goal_marker.pose.orientation
            quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
            (roll, pitch, yaw) = euler_from_quaternion(quaternion)
            rospy.loginfo(
                f"Goal Position: x={self.desired_pose.pose.position.x}, "
                f"y={self.desired_pose.pose.position.y}, z={self.desired_pose.pose.position.z}"
            )
        else:
            rospy.logwarn("No markers received in MarkerArray message")

    def task_done_callback(self, msg):
        if msg.data and self.state == 'performing_task':
            self.task_done()
            rospy.loginfo("Task done signal received. Switching back to GPS Navigation.")

    # -----------------------------
    # Timer-based transitions
    # -----------------------------

    def start_search_with_timer(self):
        # Schedules a 3-second delay before launching search
        rospy.loginfo("Will start Search for Object in 3s to avoid sudden changes...")
        self.set_pending_transition(self._search_transition_callback, 3.0)

        # Start a 30-second timer to eventually set `search_timeout_flag = True`
        self.search_timer = rospy.Timer(rospy.Duration(30), self.timer_timeout_callback, oneshot=True)

    def timer_timeout_callback(self, event):
        rospy.logwarn("Search Timeout reached. Setting flag to transition back to GPS.")
        self.search_timeout_flag = True  # Will be processed in main thread

    def _search_transition_callback(self):
        # Actually launch search node
        rospy.loginfo("Starting Search for Object now.")
        self.state_manager.transition_to_state('search_for_object')

    # This is called from main loop if `search_timeout_flag` is True
    def handle_search_timeout_flag(self):
        # This triggers the self.timeout() transition 
        # which transitions from 'search_for_object' to 'gps_navigation'
        self.search_timeout_flag = False  # reset
        self.timeout()
        rospy.loginfo("Timeout triggered. Will start GPS navigation in 3s to avoid sudden changes...")
        self.set_pending_transition(self._gps_transition_callback, 3.0)

    def _gps_transition_callback(self):
        rospy.loginfo("Starting GPS navigation now.")
        self.state_manager.transition_to_state('gps_navigation')

    def timeout(self):
        # The transitions library calls 'timeout' trigger to go to gps_navigation, 
        # but we override the actual launching with a delayed approach.
        # The 'after' transition is set to 'start_gps', which we also override. 
        pass

    # -----------------------------
    # After-transition callbacks (but with 3s delay)
    # -----------------------------

    def start_gps(self):
        rospy.loginfo("Will start GPS navigation in 3s to avoid sudden changes...")
        self.set_pending_transition(self._gps_transition_callback, 3.0)

    def start_visual_servoing(self):
        rospy.loginfo("Will start Visual Servoing in 3s to avoid sudden changes...")
        self.set_pending_transition(self._visual_servoing_transition_callback, 3.0)

    def _visual_servoing_transition_callback(self):
        rospy.loginfo("Starting Visual Servoing now.")
        self.state_manager.transition_to_state('visual_servoing')

    def perform_task(self):
        rospy.loginfo("Will start Perform Task in 3s to avoid sudden changes...")
        self.set_pending_transition(self._perform_task_callback, 3.0)

    def _perform_task_callback(self):
        rospy.loginfo("Starting Task Execution now.")
        self.state_manager.transition_to_state('performing_task')

    # -----------------------------
    # Helpers
    # -----------------------------

    def is_at_desired_pose(self, current_pose):
        dx = abs(current_pose.pose.position.x - self.desired_pose.pose.position.x)
        dy = abs(current_pose.pose.position.y - self.desired_pose.pose.position.y)
        dz = abs(current_pose.pose.position.z - self.desired_pose.pose.position.z)
        return (dx < self.desired_pose_threshold and
                dy < self.desired_pose_threshold and
                dz < self.desired_pose_threshold)

    def set_pending_transition(self, callback, delay_sec):
        """
        Schedules a callback to fire in the *main loop* after `delay_sec`.
        We'll record the time we want to run it and a reference to the callback.
        """
        self.pending_transition = callback
        self.pending_transition_time = rospy.Time.now() + rospy.Duration(delay_sec)

    # Main loop in the main thread
    def spin_main_thread(self):
        """
        Replace rospy.spin() with a custom loop so that we can 
        safely handle time-based transitions and flags in the main thread.
        """
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # 1. Check if we need to run a pending transition
            if self.pending_transition and self.pending_transition_time and now >= self.pending_transition_time:
                # It's time to do the transition
                callback = self.pending_transition
                self.pending_transition = None
                self.pending_transition_time = None
                callback()

            # 2. Check if search_timeout_flag was set
            if self.search_timeout_flag:
                self.handle_search_timeout_flag()

            rate.sleep()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize the state machine
    drone_sm = DroneStateMachine()
    # Instead of rospy.spin(), use our custom main loop
    drone_sm.spin_main_thread()
