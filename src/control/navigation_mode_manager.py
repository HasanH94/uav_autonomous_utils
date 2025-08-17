#!/usr/bin/env python3
import rospy
import numpy as np
from enum import Enum
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Bool, Float32
from nav_msgs.msg import Odometry
from uav_autonomous_utils.srv import SetNavigationMode, SetNavigationModeResponse
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
import math

class NavigationMode(Enum):
    GPS_TRACKING = "gps_tracking"
    VISUAL_SERVOING = "visual_servoing"
    SEARCH = "search"
    HOLD = "hold"
    TRANSITION = "transition"

class NavigationModeManager:
    def __init__(self):
        rospy.init_node('navigation_mode_manager')
        
        # Parameters
        self.load_parameters()
        
        # State variables
        self.current_mode = NavigationMode.HOLD
        self.previous_mode = NavigationMode.HOLD
        self.transition_progress = 0.0
        self.transition_start_time = None
        
        # Position and target tracking
        self.current_pose = None
        self.current_velocity = None
        self.gps_target = None
        self.visual_target = None
        self.active_target = None
        self.target_distance = float('inf')
        
        # ArUco detection state
        self.aruco_detected = False
        self.aruco_confidence = 0.0
        self.last_aruco_time = rospy.Time(0)
        
        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.setup_publishers()
        
        # Subscribers
        self.setup_subscribers()
        
        # Services
        self.mode_service = rospy.Service('/navigation/set_mode', SetNavigationMode, self.handle_set_mode)
        
        # Main control timer
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_rate), self.control_callback)
        
        rospy.loginfo("Navigation Mode Manager initialized")
        
    def load_parameters(self):
        # Control parameters
        self.control_rate = rospy.get_param('~control_rate', 20.0)
        self.transition_duration = rospy.get_param('~transition_duration', 2.0)
        
        # Visual servoing parameters
        self.visual_activation_distance = rospy.get_param('~visual_activation_distance', 10.0)
        self.visual_confidence_threshold = rospy.get_param('~visual_confidence_threshold', 0.7)
        self.aruco_timeout = rospy.get_param('~aruco_timeout', 2.0)
        
        # Mode switching parameters
        self.auto_switch_modes = rospy.get_param('~auto_switch_modes', True)
        self.hold_on_target_loss = rospy.get_param('~hold_on_target_loss', False)
        
    def setup_publishers(self):
        # Mode and status publishers
        self.mode_pub = rospy.Publisher('/navigation/current_mode', String, queue_size=1)
        self.target_distance_pub = rospy.Publisher('/navigation/target_distance', Float32, queue_size=1)
        
        # Control output publishers
        self.goal_output_pub = rospy.Publisher('/navigation/active_goal', PoseStamped, queue_size=10)
        
        # Remapped goal publishers for different controllers
        self.gps_goal_pub = rospy.Publisher('/move_base_gps', PoseStamped, queue_size=10)
        self.visual_goal_pub = rospy.Publisher('/move_base_visual', PoseStamped, queue_size=10)
        
    def setup_subscribers(self):
        # Odometry and state
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        
        # Goal inputs
        rospy.Subscriber('/move_base_mission', PoseStamped, self.gps_goal_callback)
        rospy.Subscriber('/aruco_offset_pose', PoseStamped, self.visual_goal_callback)
        
        # ArUco detection status
        rospy.Subscriber('/aruco_detection_status', Bool, self.aruco_status_callback)
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist
        
        # Update target distance if we have an active target
        if self.active_target:
            self.update_target_distance()
            
    def gps_goal_callback(self, msg):
        self.gps_target = msg
        rospy.loginfo(f"GPS target updated: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")
        
        # Auto-switch to GPS mode if configured
        if self.auto_switch_modes and self.current_mode == NavigationMode.HOLD:
            self.switch_mode(NavigationMode.GPS_TRACKING)
            
    def visual_goal_callback(self, msg):
        self.visual_target = msg
        self.last_aruco_time = rospy.Time.now()
        
    def aruco_status_callback(self, msg):
        self.aruco_detected = msg.data
        if msg.data:
            self.aruco_confidence = min(1.0, self.aruco_confidence + 0.1)
        else:
            self.aruco_confidence = max(0.0, self.aruco_confidence - 0.05)
            
    def update_target_distance(self):
        if not self.current_pose or not self.active_target:
            return
            
        dx = self.active_target.pose.position.x - self.current_pose.position.x
        dy = self.active_target.pose.position.y - self.current_pose.position.y
        dz = self.active_target.pose.position.z - self.current_pose.position.z
        
        self.target_distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.target_distance_pub.publish(Float32(self.target_distance))
        
    def switch_mode(self, new_mode):
        if new_mode == self.current_mode:
            return
            
        rospy.loginfo(f"Switching from {self.current_mode.value} to {new_mode.value}")
        
        # Start transition if needed
        if self.transition_duration > 0:
            self.previous_mode = self.current_mode
            self.current_mode = NavigationMode.TRANSITION
            self.transition_start_time = rospy.Time.now()
            self.transition_target_mode = new_mode
        else:
            self.previous_mode = self.current_mode
            self.current_mode = new_mode
            
        self.mode_pub.publish(String(self.current_mode.value))
        
    def handle_set_mode(self, req):
        try:
            new_mode = NavigationMode(req.mode)
            self.switch_mode(new_mode)
            return SetNavigationModeResponse(success=True, message=f"Switched to {new_mode.value}")
        except ValueError:
            return SetNavigationModeResponse(success=False, message=f"Invalid mode: {req.mode}")
            
    def check_auto_mode_switch(self):
        """Automatically switch between GPS and Visual modes based on conditions"""
        if not self.auto_switch_modes:
            return
            
        # Check for visual servoing activation
        if (self.current_mode == NavigationMode.GPS_TRACKING and 
            self.aruco_detected and 
            self.aruco_confidence > self.visual_confidence_threshold and
            self.target_distance < self.visual_activation_distance):
            
            rospy.loginfo("Auto-switching to VISUAL_SERVOING mode")
            self.switch_mode(NavigationMode.VISUAL_SERVOING)
            
        # Check for fallback to GPS
        elif (self.current_mode == NavigationMode.VISUAL_SERVOING and
              (rospy.Time.now() - self.last_aruco_time).to_sec() > self.aruco_timeout):
            
            rospy.loginfo("Lost visual target, switching back to GPS_TRACKING")
            self.switch_mode(NavigationMode.GPS_TRACKING)
            
    def control_callback(self, event):
        """Main control loop"""
        
        # Handle mode transitions
        if self.current_mode == NavigationMode.TRANSITION:
            self.handle_transition()
            
        # Check for automatic mode switching
        self.check_auto_mode_switch()
        
        # Select active target based on mode
        if self.current_mode == NavigationMode.GPS_TRACKING:
            self.active_target = self.gps_target
        elif self.current_mode == NavigationMode.VISUAL_SERVOING:
            self.active_target = self.visual_target if self.visual_target else self.gps_target
        elif self.current_mode == NavigationMode.SEARCH:
            # In search mode, we might want to hover or follow a search pattern
            self.active_target = None
        else:
            self.active_target = None
            
        # Publish active goal
        if self.active_target:
            # The manager now simply forwards the active target to the appropriate topic.
            # The PID controller handles all yaw logic.
            if self.current_mode == NavigationMode.GPS_TRACKING:
                self.gps_goal_pub.publish(self.active_target)
            elif self.current_mode == NavigationMode.VISUAL_SERVOING:
                self.visual_goal_pub.publish(self.active_target)
                
            # Also publish to general output for monitoring
            self.goal_output_pub.publish(self.active_target)
            
        # Publish current states
        self.mode_pub.publish(String(self.current_mode.value))
        
    def handle_transition(self):
        """Handle smooth transition between modes"""
        if not self.transition_start_time:
            return
            
        elapsed = (rospy.Time.now() - self.transition_start_time).to_sec()
        self.transition_progress = min(1.0, elapsed / self.transition_duration)
        
        if self.transition_progress >= 1.0:
            # Transition complete
            self.current_mode = self.transition_target_mode
            self.transition_start_time = None
            rospy.loginfo(f"Transition complete, now in {self.current_mode.value}")
            
    def shutdown(self):
        rospy.loginfo("Navigation Mode Manager shutting down")
        self.control_timer.shutdown()
        
if __name__ == '__main__':
    try:
        manager = NavigationModeManager()
        rospy.on_shutdown(manager.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass