#!/usr/bin/env python3
import rospy
import roslaunch
import subprocess
from transitions import Machine
from threading import Timer
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import math


class StateManager:
    """
    This class is responsible for launching/stopping the actual ROS processes
    (like local_node.launch, PID node, search node, etc.).
    Customize these as needed.
    """
    def __init__(self):
        self.current_process = None

    def launch_file(self, package, launch_file):
        """Launch a .launch file (e.g., local_node.launch)."""
        rospy.loginfo(f"Attempting to launch file: {package}/{launch_file}")
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
            rospy.loginfo(f"Launched: {package}/{launch_file}")
        except Exception as e:
            rospy.logerr(f"Failed to launch {package}/{launch_file}: {e}")

    def run_node(self, package, executable):
        """Use rosrun to start a single node (Python script or otherwise)."""
        rospy.loginfo(f"Attempting to run node: {package}/{executable}")
        self.stop_current_process()
        try:
            command = f"rosrun {package} {executable}"
            self.current_process = subprocess.Popen(command, shell=True)
            rospy.loginfo(f"Started node via: {command}")
        except Exception as e:
            rospy.logerr(f"Failed to run node {package}/{executable}: {e}")

    # def stop_current_process(self):
    #     """Stop any currently running process or launch file."""
    #     if self.current_process:
    #         rospy.loginfo(f"Stopping current process: {self.current_process}")
    #         try:
    #             if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
    #                 self.current_process.shutdown()
    #                 self.current_process.join()  # Ensure shutdown completes
    #                 rospy.loginfo("roslaunch parent shutdown complete.")
    #             else:
    #                 self.current_process.terminate()
    #                 self.current_process.wait()
    #                 rospy.loginfo("Terminated subprocess.")
    #             rospy.loginfo("Stopped current process.")
    #         except Exception as e:
    #             rospy.logerr(f"Error stopping process: {e}")
    #         finally:
    #             self.current_process = None
    #     else:
    #         rospy.loginfo("No current process to stop.")
    def stop_current_process(self):
        """Stop any currently running process or launch file."""
        if self.current_process:
            rospy.loginfo(f"Stopping current process: {self.current_process}")
            try:
                if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
                    self.current_process.shutdown()
                    rospy.loginfo("roslaunch parent shutdown complete.")
                else:
                    self.current_process.terminate()
                    self.current_process.wait()
                    rospy.loginfo("Terminated subprocess.")
                rospy.loginfo("Stopped current process.")
            except Exception as e:
                rospy.logerr(f"Error stopping process: {e}")
            finally:
                self.current_process = None
        else:
            rospy.loginfo("No current process to stop.")




# ------------------------------------------------------------------------

class DroneStateMachine:
    states = [
        'gps_navigation',
        'visual_servoing',
        'performing_task',
        'search_for_object'
    ]
    
    def __init__(self):
        # Initialize manager that launches/stops processes
        self.manager = StateManager()

        # Initialize goal and current positions
        self.current_position = None
        self.goal_position = None
        self.threshold = 0.2  # 0.2 meters

        # Build the transitions machine
        self.machine = Machine(
            model=self,
            states=DroneStateMachine.states,
            initial='gps_navigation',
            auto_transitions=False  # Disable automatic transitions
        )

        # Add transitions matching your XState machine:
        self.machine.add_transition(
            trigger='REACHED_GOAL_POINT',
            source='gps_navigation',
            dest='visual_servoing',
            after='logVisualServoingStart'
        )
        self.machine.add_transition(
            trigger='REACHED_DESIRED_POSE',
            source='visual_servoing',
            dest='performing_task',
            after='logTaskStart'
        )
        self.machine.add_transition(
            trigger='OBJECT_NOT_FOUND',
            source='visual_servoing',
            dest='search_for_object',
            after='logSearchStart'
        )
        self.machine.add_transition(
            trigger='TASK_DONE',
            source='performing_task',
            dest='gps_navigation',
            after='logGpsStart'
        )
        self.machine.add_transition(
            trigger='OBJECT_FOUND',
            source='search_for_object',
            dest='visual_servoing',
            after='logVisualServoingStart'
        )
        self.machine.add_transition(
            trigger='TIMEOUT',
            source='search_for_object',
            dest='gps_navigation',
            after='logGpsStart'
        )

        # Initialize Timer
        self.search_timer = None

        # Start with GPS navigation
        self.logGpsStart()

        # Initialize ROS Subscribers
        self._init_subscribers()

    def _init_subscribers(self):
        """Initialize ROS subscribers for state transitions and position monitoring."""
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.loginfo("Subscribers initialized and listening to /goal_position and /mavros/local_position/pose topics.")

    def goal_callback(self, msg):
        """Callback to update the goal position from /goal_position."""
        if not msg.markers:
            rospy.logwarn("Received empty MarkerArray on /goal_position.")
            return

        # Assuming the first marker contains the goal position
        marker = msg.markers[0]
        self.goal_position = marker.pose.position
        rospy.loginfo(f"Received new goal position: x={self.goal_position.x}, y={self.goal_position.y}, z={self.goal_position.z}")

    # def position_callback(self, msg):
    #     """Callback to monitor drone's current position and trigger state transitions."""
    #     self.current_position = msg.pose.position
    #     if self.goal_position is None:
    #         # No goal set yet
    #         return
        
    #     distance = self.calculate_distance(self.current_position, self.goal_position)
    #     rospy.logdebug(f"Current Position: x={self.current_position.x}, y={self.current_position.y}, z={self.current_position.z}")
    #     rospy.logdebug(f"Goal Position: x={self.goal_position.x}, y={self.goal_position.y}, z={self.goal_position.z}")
    #     rospy.logdebug(f"Distance to Goal: {distance} meters")

    #     if distance <= self.threshold:
    #         rospy.loginfo("Drone has reached the goal position within the threshold.")
    #         # Reset goal_position to prevent multiple triggers for the same goal
    #         self.goal_position = None
    #         rospy.loginfo("Triggering REACHED_GOAL_POINT")
    #         self.REACHED_GOAL_POINT()
    def position_callback(self, msg):
        """Callback to monitor drone's current position and trigger state transitions."""
        self.current_position = msg.pose.position
        if self.goal_position is None:
            # No goal set yet
            return
        
        distance = self.calculate_distance(self.current_position, self.goal_position)
        rospy.logdebug(f"Current Position: x={self.current_position.x}, y={self.current_position.y}, z={self.current_position.z}")
        rospy.logdebug(f"Goal Position: x={self.goal_position.x}, y={self.goal_position.y}, z={self.goal_position.z}")
        rospy.logdebug(f"Distance to Goal: {distance} meters")

        # Only trigger if in 'gps_navigation' state
        if distance <= self.threshold and self.state == 'gps_navigation':
            rospy.loginfo("Drone has reached the goal position within the threshold.")
            # Reset goal_position to prevent multiple triggers for the same goal
            self.goal_position = None
            rospy.loginfo("Triggering REACHED_GOAL_POINT")
            self.REACHED_GOAL_POINT()


    @staticmethod
    def calculate_distance(pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )

    # -------------
    # Actions from your XState "actions" block
    # -------------

    def logVisualServoingStart(self):
        rospy.loginfo("Switching to Visual Servoing")
        # Stop any current node, then run your PID node or whatever
        self.manager.run_node('othmanPack', 'PID.py')

    def logTaskStart(self):
        rospy.loginfo("Reached Desired Pose, Performing Task")
        # Launch or run your "perform_task_node.py"
        self.manager.run_node('othmanPack', 'perform_task_node.py')

    def logSearchStart(self):
        rospy.loginfo("Switching to Search for Object")
        # Launch or run your "search_for_object_node.py"
        self.manager.run_node('othmanPack', 'search_for_object_node.py')
        # Start 30-second timer
        self.start_search_timer()

    def logGpsStart(self):
        rospy.loginfo("Starting GPS navigation")
        # Launch your "local_node.launch"
        self.manager.launch_file('local_planner', 'local_node.launch')

    # -------------
    # Timer Handling for search_for_object
    # -------------

    def start_search_timer(self):
        """Start a 30-second timer for the search_for_object state."""
        # Cancel any existing timer first
        if self.search_timer is not None:
            self.search_timer.cancel()
            rospy.loginfo("Existing search timer canceled.")

        # Start 30s timer
        self.search_timer = Timer(30.0, self._on_search_timeout)
        self.search_timer.start()
        rospy.loginfo("30-second timer started for search_for_object state.")

    def on_enter_search_for_object(self):
        """
        Called automatically by transitions whenever we move into 'search_for_object'.
        Start a 30-second Timer.
        """
        rospy.loginfo("Entering search_for_object state.")
        self.start_search_timer()

    def on_exit_search_for_object(self):
        """
        Called automatically by transitions whenever we leave 'search_for_object'.
        Cancel the timer to prevent it from firing.
        """
        if self.search_timer is not None:
            self.search_timer.cancel()
            self.search_timer = None
            rospy.loginfo("Exited search_for_object state. Timer canceled.")

    def _on_search_timeout(self):
        """
        If we do not transition out of search_for_object within 30 seconds,
        trigger TIMEOUT to transition to gps_navigation.
        """
        rospy.logwarn("30-second search_for_object timer expired. Triggering TIMEOUT.")
        if self.state == 'search_for_object':
            self.TIMEOUT()

    # No duplicate methods


# -------------------------------------------------------------------------------

if __name__ == '__main__':
    rospy.init_node('drone_state_machine_node', anonymous=True, log_level=rospy.INFO)
    sm = DroneStateMachine()
    rospy.spin()
