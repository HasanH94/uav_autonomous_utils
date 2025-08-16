#!/usr/bin/env python3
from transitions import Machine
import rospy
from std_msgs.msg import Bool  # Assuming object detection is a Boolean
from geometry_msgs.msg import PoseStamped  # Example for position checking
from visualization_msgs import MarkerArray
from tf.transformations import euler_from_quaternion

class DroneStateMachine:
    states = ['gps_navigation', 'visual_servoing', 'performing_task','search_for_object']

    def __init__(self):
        # Initialize state machine
        self.machine = Machine(model=self, states=DroneStateMachine.states, initial='gps_navigation')
        self.desired_pose = PoseStamped()
        self.desired_pose_threshold = 0.1  # Adjust threshold as needed


        # Define transitions
        self.machine.add_transition(trigger='object_detected', source='gps_navigation', dest='visual_servoing')
        self.machine.add_transition(trigger='reached_desired_pose', source='visual_servoing', dest='performing_task')
        self.machine.add_transition(trigger='object_lost', source='visual_servoing', dest='search_for_object')
        self.machine.add_transition(trigger='object_found',source='searching_for_object', dest = 'visual_servoing')
        self.machine.add_transition(trigger='task_done', source='performing_task', dest='gps_navigation')

        # ROS Publishers and Subscribers
        rospy.Subscriber('/marker_detection_status', Bool, self.detection_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/goal_position', MarkerArray, self.goal_callback)
        
        self.gps_publisher = rospy.Publisher('/gps_navigation_command', PoseStamped, queue_size=10)
        self.visual_servo_publisher = rospy.Publisher('/visual_servo_command', PoseStamped, queue_size=10)


    def detection_callback(self, msg):
        if msg.data:  # Object detected
            if self.state == 'gps_navigation':
                self.object_detected()
                rospy.loginfo("Switching to Visual Servoing")
                self.stop_gps()
                self.start_visual_servoing()
        else:  # Object lost
            if self.state == 'visual_servoing':
                self.object_lost()
                rospy.loginfo("Switching to GPS Navigation")
                self.stop_visual_servoing()
                self.start_gps()

    def pose_callback(self, pose_msg):
        # Check if the drone is within the desired pose threshold
        if self.state == 'visual_servoing' and self.is_at_desired_pose(pose_msg):
            self.reached_desired_pose()
            rospy.loginfo("Reached Desired Pose, Performing Task")
            self.stop_visual_servoing()
            self.perform_task()

    def goal_callback(self, msg):
        # Assuming we're interested in the first marker in the array
        if msg.markers:
            goal_marker = msg.markers[0]
            self.desired_pose.x = goal_marker.pose.position.x
            self.desired_pose.y = goal_marker.pose.position.y
            self.desired_pose.z = goal_marker.pose.position.z
            orientation_q = goal_marker.pose.orientation
            quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
            (self.desired_pose.roll, self.desired_pose.pitch, self.desired_pose.yaw) = euler_from_quaternion(quaternion)

            rospy.loginfo(f"Goal Position: x={self.desired_pose.x}, y={self.desired_pose.y}, z={self.desired_pose.z}")
        else:
            rospy.logwarn("No markers received in MarkerArray message")

    def is_at_desired_pose(self, pose):
        # Check if current pose is within the desired pose threshold
        return abs(pose.pose.position.x - self.desired_pose.pose.position.x) < self.desired_pose_threshold and \
               abs(pose.pose.position.y - self.desired_pose.pose.position.y) < self.desired_pose_threshold and \
               abs(pose.pose.position.z - self.desired_pose.pose.position.z) < self.desired_pose_threshold

    def stop_gps(self):
        # Stop GPS-based navigation (implementation example)
        rospy.loginfo("Stopping GPS navigation")
        self.gps_publisher.publish(PoseStamped())  # Publishing empty message or stop command

    def start_gps(self):
        # Start GPS-based navigation (implementation example)
        rospy.loginfo("Starting GPS navigation")
        gps_pose = PoseStamped()
        gps_pose.pose.position.z = 1.5  # Example altitude
        self.gps_publisher.publish(gps_pose)

    def stop_visual_servoing(self):
        # Stop visual servoing (implementation example)
        rospy.loginfo("Stopping Visual Servoing")
        self.visual_servo_publisher.publish(PoseStamped())  # Publishing empty message or stop command

    def start_visual_servoing(self):
        # Start visual servoing (implementation example)
        rospy.loginfo("Starting Visual Servoing")
        servo_pose = PoseStamped()
        servo_pose.pose.position.x = self.desired_pose.pose.position.x
        servo_pose.pose.position.y = self.desired_pose.pose.position.y
        servo_pose.pose.position.z = self.desired_pose.pose.position.z
        self.visual_servo_publisher.publish(servo_pose)

    def perform_task(self):
        # Code to execute when drone is in the performing task state
        rospy.loginfo("Performing Task")
        rospy.sleep(4)  # Simulate task time
        self.task_done()

if __name__ == "__main__":
    rospy.init_node('drone_state_machine')
    drone_sm = DroneStateMachine()
    rospy.spin()
