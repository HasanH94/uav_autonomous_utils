#!/usr/bin/env python3
from transitions import Machine
import rospy
from std_msgs.msg import Bool  # Assuming object detection is a Boolean
from geometry_msgs.msg import PoseStamped  # Example for position checking
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
import roslaunch
import subprocess

class StateManager:
    def __init__(self):
        self.current_process = None
        rospy.init_node('state_manager_node', anonymous=True)

    def launch_file(self, package, launch_file):
        if self.current_process:
            self.stop_current_process()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]
        self.current_process = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        self.current_process.start()
        rospy.loginfo(f"Started launch file: {package}/{launch_file}")

    def run_node(self, package, node_type, node_name):
        if self.current_process:
            self.stop_current_process()
        command = f"rosrun {package} {node_type}"
        self.current_process = subprocess.Popen(command, shell=True)
        rospy.loginfo(f"Started node: {package}/{node_type}")

    def stop_current_process(self):
        if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
            self.current_process.shutdown()
        elif isinstance(self.current_process, subprocess.Popen):
            self.current_process.terminate()
            self.current_process.wait()
        self.current_process = None
        rospy.loginfo("Stopped current process")

    def transition_to_state(self, state_name):
        if state_name == 'gps_navigation':
            self.launch_file('my_package', 'gps_navigation.launch')
        elif state_name == 'visual_servoing':
            self.run_node('my_package', 'visual_servoing_node', 'visual_servoing')
        elif state_name == 'performing_task':
            self.run_node('my_package', 'perform_task_node', 'perform_task')
        elif state_name == 'search_for_object':
            self.run_node('my_package', 'search_object_node', 'search_for_object')
        else:
            rospy.logwarn(f"Unknown state: {state_name}")

class DroneStateMachine:
    states = ['gps_navigation', 'visual_servoing', 'performing_task', 'search_for_object']

    def __init__(self):
        # Initialize the state machine
        self.machine = Machine(model=self, states=DroneStateMachine.states, initial='gps_navigation')
        self.state_manager = StateManager()  # Integrate StateManager

        # Define transitions with associated callbacks
        self.machine.add_transition(trigger='object_detected', source='gps_navigation', dest='visual_servoing', after='start_visual_servoing')
        self.machine.add_transition(trigger='reached_desired_pose', source='visual_servoing', dest='performing_task', after='perform_task')
        self.machine.add_transition(trigger='object_lost', source='visual_servoing', dest='search_for_object', after='search_for_object')
        self.machine.add_transition(trigger='object_found', source='search_for_object', dest='visual_servoing', after='start_visual_servoing')
        self.machine.add_transition(trigger='task_done', source='performing_task', dest='gps_navigation', after='start_gps')

        # ROS Subscribers
        rospy.Subscriber('/marker_detection_status', Bool, self.detection_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)

        self.desired_pose = PoseStamped()
        self.desired_pose_threshold = 0.1  # Adjust threshold as needed

    def detection_callback(self, msg):
        if msg.data:  # Object detected
            if self.state == 'gps_navigation':
                self.object_detected()
                rospy.loginfo("Switching to Visual Servoing")
        else:  # Object lost
            if self.state == 'visual_servoing':
                self.object_lost()
                rospy.loginfo("Switching to Search for Object")

    def pose_callback(self, pose_msg):
        # Check if the drone is within the desired pose threshold
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
            (self.desired_pose.roll, self.desired_pose.pitch, self.desired_pose.yaw) = euler_from_quaternion(quaternion)
            rospy.loginfo(f"Goal Position: x={self.desired_pose.pose.position.x}, y={self.desired_pose.pose.position.y}, z={self.desired_pose.pose.position.z}")
        else:
            rospy.logwarn("No markers received in MarkerArray message")

    def is_at_desired_pose(self, pose):
        return abs(pose.pose.position.x - self.desired_pose.pose.position.x) < self.desired_pose_threshold and \
               abs(pose.pose.position.y - self.desired_pose.pose.position.y) < self.desired_pose_threshold and \
               abs(pose.pose.position.z - self.desired_pose.pose.position.z) < self.desired_pose_threshold

    # Replace individual state methods with StateManager calls
    def start_gps(self):
        rospy.loginfo("Starting GPS navigation")
        self.state_manager.transition_to_state('gps_navigation')

    def start_visual_servoing(self):
        rospy.loginfo("Starting Visual Servoing")
        self.state_manager.transition_to_state('visual_servoing')

    def perform_task(self):
        rospy.loginfo("Performing Task")
        self.state_manager.transition_to_state('performing_task')

    def search_for_object(self):
        rospy.loginfo("Searching for Object")
        self.state_manager.transition_to_state('search_for_object')

if __name__ == "__main__":
    drone_sm = DroneStateMachine()
    rospy.spin()
