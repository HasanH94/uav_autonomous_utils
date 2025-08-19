#!/usr/bin/env python3
import rospy
import math
from transitions import Machine
from threading import Timer, Lock
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String, Float32
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool
from mavros_msgs.msg import State
from tf.transformations import euler_from_quaternion

class EnhancedDroneStateMachine(Machine):
    """
    Enhanced State Machine that uses Navigation Mode Manager
    instead of launching/killing nodes
    """
    
    states = [
        {'name': 'idle', 'on_enter': 'on_enter_idle'},
        {'name': 'gps_navigation', 'on_enter': 'on_enter_gps_navigation', 'on_exit': 'on_exit_gps_navigation'},
        {'name': 'visual_servoing', 'on_enter': 'on_enter_visual_servoing', 'on_exit': 'on_exit_visual_servoing'},
        {'name': 'search_for_object', 'on_enter': 'on_enter_search_for_object', 'on_exit': 'on_exit_search_for_object'},
        {'name': 'performing_task', 'on_enter': 'on_enter_performing_task', 'on_exit': 'on_exit_performing_task'},
        {'name': 'returning_home', 'on_enter': 'on_enter_returning_home', 'on_exit': 'on_exit_returning_home'},
        {'name': 'landing', 'on_enter': 'on_enter_landing'},
    ]
    
    def __init__(self):
        super().__init__(
            model=self,
            states=EnhancedDroneStateMachine.states,
            initial='idle',
            auto_transitions=False,
            send_event=True
        )
        
        # Parameters
        self.goal_points = rospy.get_param('~goal_points', [])
        self.home_position = rospy.get_param('~home_position', [0.0, 0.0, 2.0])
        
        # ArUco detection with sliding window
        self.aruco_detection_window_size = rospy.get_param('~aruco_detection_window_size', 30)  # frames (~1.5s at 20Hz)
        self.aruco_detection_threshold = rospy.get_param('~aruco_detection_threshold', 0.7)  # 70% of frames must have detection
        self.aruco_detection_history = []  # Sliding window of detections (True/False)
        
        self.visual_activation_distance = rospy.get_param('~visual_activation_distance', 15.0)
        self.gps_goal_tolerance = rospy.get_param('~gps_goal_tolerance', 1.0)  # Consider reached if within 1m
        self.visual_gps_tolerance = rospy.get_param('~visual_gps_tolerance', 3.0) # Must be near GPS goal to activate visual
        self.search_timeout = rospy.get_param('~search_timeout', 30.0)
        self.visual_pos_tolerance = rospy.get_param('~visual_pos_tolerance', 0.2)
        self.visual_yaw_tolerance_deg = rospy.get_param('~visual_yaw_tolerance_deg', 5.0)

        # State tracking
        self.current_mission_goal_index = 0
        self.aruco_detected_consistent = False  # True when detection is consistent enough
        self.target_distance = float('inf')
        self.navigation_mode = "hold"
        self.search_timer = None
        self.current_pose = None  # Initialize pose
        self.visual_target_pose = None # Store visual target pose
        self.mavros_state = State()  # Initialize state
        
        # Goal management (from navigation_mode_manager)
        self.gps_target = None
        self.visual_target = None
        self.active_target = None
        
        # Services (removed navigation/set_mode since we're the authority now)
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        
        self.mavros_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

        # Periodic check for goal status
        self.goal_check_timer = rospy.Timer(rospy.Duration(0.1), self.check_goal_status)
        
        # Publishers
        self.mission_goal_pub = rospy.Publisher('/move_base_mission', PoseStamped, queue_size=10)
        self.state_pub = rospy.Publisher('/state_machine/current_state', String, queue_size=1)
        
        # Navigation mode and goal publishers (replacing navigation_mode_manager)
        self.mode_pub = rospy.Publisher('/navigation/current_mode', String, queue_size=1)
        self.target_distance_pub = rospy.Publisher('/navigation/target_distance', Float32, queue_size=1)
        self.gps_goal_pub = rospy.Publisher('/move_base_gps', PoseStamped, queue_size=10)
        self.visual_goal_pub = rospy.Publisher('/move_base_visual', PoseStamped, queue_size=10)
        
        # Subscribers
        self.setup_subscribers()
        
        # Define transitions
        self.setup_transitions()
        
        rospy.loginfo("Enhanced Drone State Machine initialized")
        
    def setup_subscribers(self):
        # ArUco detection and visual target
        rospy.Subscriber('/aruco_detection_status', Bool, self.aruco_status_callback)
        rospy.Subscriber('/aruco_offset_pose', PoseStamped, self.visual_goal_callback)
        
        # Event triggers
        # rospy.Subscriber('/drone_events/reached_goal_point', Bool, self.goal_reached_callback) # Removed, now handled internally
        rospy.Subscriber('/drone_events/task_done', Bool, self.task_done_callback)
        
        # MAVROS state
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        
    def setup_transitions(self):
        # From idle
        self.add_transition(
            trigger='start_mission',
            source='idle',
            dest='gps_navigation',
            conditions=['has_mission_goals']
        )
        
        # From GPS navigation
        self.add_transition(
            trigger='visual_target_acquired',
            source='gps_navigation',
            dest='visual_servoing',
            conditions=['is_aruco_detected_confident', 'is_within_visual_range', 'is_near_gps_goal']
        )
        
        # GPS goal reached (within 1m) but no ArUco - start searching
        self.add_transition(
            trigger='gps_goal_reached',
            source='gps_navigation',
            dest='search_for_object',
            unless=['is_aruco_detected_confident']
        )
        
        # From visual servoing
        self.add_transition(
            trigger='visual_goal_reached',
            source='visual_servoing',
            dest='performing_task'
        )
        
        self.add_transition(
            trigger='visual_target_lost',
            source='visual_servoing',
            dest='search_for_object'
        )
        
        # From search
        self.add_transition(
            trigger='target_found',
            source='search_for_object',
            dest='visual_servoing',
            conditions=['is_aruco_detected_confident']
        )
        
        # Search timeout - skip to next GPS waypoint
        self.add_transition(
            trigger='search_timeout_triggered',
            source='search_for_object',
            dest='gps_navigation',
            before='skip_to_next_goal'  # Skip current goal, move to next
        )
        
        # From performing task
        self.add_transition(
            trigger='task_completed',
            source='performing_task',
            dest='gps_navigation',
            unless=['is_last_goal'],
            before='increment_goal_index'
        )
        
        self.add_transition(
            trigger='task_completed',
            source='performing_task',
            dest='returning_home',
            conditions=['is_last_goal']
        )
        
        # From returning home
        self.add_transition(
            trigger='home_reached',
            source='returning_home',
            dest='landing'
        )
        
    # Condition checks
    def has_mission_goals(self, event=None):
        return len(self.goal_points) > 0
        
    def is_aruco_detected_confident(self, event=None):
        return self.aruco_detected_consistent
        
    def is_within_visual_range(self, event=None):
        return self.target_distance <= self.visual_activation_distance
        
    def is_last_goal(self, event=None):
        return self.current_mission_goal_index >= len(self.goal_points) - 1

    def is_near_gps_goal(self, event=None):
        if not self.current_pose or self.current_mission_goal_index >= len(self.goal_points):
            return False
        
        goal_coords = self.goal_points[self.current_mission_goal_index]
        dx = goal_coords[0] - self.current_pose.pose.position.x
        dy = goal_coords[1] - self.current_pose.pose.position.y
        dz = goal_coords[2] - self.current_pose.pose.position.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance <= self.visual_gps_tolerance

    def check_visual_goal_reached_conditions(self):
        if not self.current_pose or not self.visual_target_pose:
            rospy.loginfo_throttle(1.0, f"Visual Goal Check: current_pose or visual_target_pose is None. current_pose: {self.current_pose is not None}, visual_target_pose: {self.visual_target_pose is not None}")
            return False
        
        rospy.loginfo_throttle(2.0, f"DEBUG: Checking visual goal - State: {self.state}, ArUco confident: {self.aruco_detected_consistent}, Distance: {self.target_distance:.2f}m")

        # Position check
        dx = self.visual_target_pose.pose.position.x - self.current_pose.pose.position.x
        dy = self.visual_target_pose.pose.position.y - self.current_pose.pose.position.y
        dz = self.visual_target_pose.pose.position.z - self.current_pose.pose.position.z
        pos_distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        pos_reached = pos_distance <= self.visual_pos_tolerance

        # Yaw check
        q_current = self.current_pose.pose.orientation
        q_target = self.visual_target_pose.pose.orientation

        _, _, current_yaw = euler_from_quaternion([q_current.x, q_current.y, q_current.z, q_current.w])
        _, _, target_yaw = euler_from_quaternion([q_target.x, q_target.y, q_target.z, q_target.w])

        yaw_error = target_yaw - current_yaw
        # Normalize yaw error to -pi to pi
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        yaw_error_deg = math.degrees(abs(yaw_error))
        yaw_reached = yaw_error_deg <= self.visual_yaw_tolerance_deg

        rospy.loginfo_throttle(1.0, f"Visual Goal Check: pos_dist={pos_distance:.2f}m (tol={self.visual_pos_tolerance:.2f}m), yaw_err={yaw_error_deg:.2f}deg (tol={self.visual_yaw_tolerance_deg:.2f}deg)")

        if pos_reached and yaw_reached:
            rospy.loginfo(f"Visual goal reached: pos_dist={pos_distance:.2f}m (tol={self.visual_pos_tolerance:.2f}m), yaw_err={yaw_error_deg:.2f}deg (tol={self.visual_yaw_tolerance_deg:.2f}deg)")
        
        return pos_reached and yaw_reached

    def check_gps_goal_reached_conditions(self):
        if self.state == "gps_navigation":
            if self.current_pose and self.current_mission_goal_index < len(self.goal_points):
                goal = self.goal_points[self.current_mission_goal_index]
                dx = goal[0] - self.current_pose.pose.position.x
                dy = goal[1] - self.current_pose.pose.position.y
                dz = goal[2] - self.current_pose.pose.position.z
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)

                if distance <= self.gps_goal_tolerance:
                    rospy.loginfo(f"GPS goal reached (within {distance:.2f}m)")
                    return True
        return False

    # Callbacks
    def visual_goal_callback(self, msg):
        """Handle visual target from ArUco detection"""
        self.visual_target = msg
        self.visual_target_pose = msg  # Keep for compatibility
        
        # Update target distance if we have a pose
        if self.current_pose:
            self.update_target_distance()
    
    def update_target_distance(self):
        """Calculate distance to active target"""
        if not self.current_pose:
            return
            
        # Determine which target to measure distance to
        target = None
        if self.state == "visual_servoing" and self.visual_target:
            target = self.visual_target
        elif self.state == "gps_navigation" and self.gps_target:
            target = self.gps_target
            
        if target:
            dx = target.pose.position.x - self.current_pose.pose.position.x
            dy = target.pose.position.y - self.current_pose.pose.position.y
            dz = target.pose.position.z - self.current_pose.pose.position.z
            self.target_distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            self.target_distance_pub.publish(Float32(self.target_distance))
        
        # Check for visual activation
        if (self.state == "gps_navigation" and 
            self.is_aruco_detected_confident() and 
            self.is_within_visual_range() and
            self.is_near_gps_goal()):
            rospy.loginfo(f"Visual activation check - ArUco: {self.is_aruco_detected_confident()}, Range: {self.is_within_visual_range()}, Near GPS: {self.is_near_gps_goal()}")
            self.visual_target_acquired()
            
    def aruco_status_callback(self, msg):
        # Add current detection to sliding window
        self.aruco_detection_history.append(msg.data)
        
        # Maintain window size
        if len(self.aruco_detection_history) > self.aruco_detection_window_size:
            self.aruco_detection_history.pop(0)  # Remove oldest
            
        # Calculate detection rate in the window
        if len(self.aruco_detection_history) >= self.aruco_detection_window_size:
            detection_rate = sum(self.aruco_detection_history) / len(self.aruco_detection_history)
            
            # Check if detection is consistent enough
            was_consistent = self.aruco_detected_consistent
            self.aruco_detected_consistent = detection_rate >= self.aruco_detection_threshold
            
            # Log significant changes
            if not was_consistent and self.aruco_detected_consistent:
                rospy.loginfo(f"ArUco detection CONSISTENT: {detection_rate:.1%} over {len(self.aruco_detection_history)} frames")
                
                # Trigger state transitions if appropriate
                if self.state == "search_for_object":
                    self.target_found()
                elif self.state == "gps_navigation" and self.is_within_visual_range() and self.is_near_gps_goal():
                    rospy.loginfo(f"ArUco callback: Attempting visual transition - Range: {self.is_within_visual_range()}, Near GPS: {self.is_near_gps_goal()}")
                    self.visual_target_acquired()
                    
            elif was_consistent and not self.aruco_detected_consistent:
                rospy.loginfo(f"ArUco detection LOST: {detection_rate:.1%} over {len(self.aruco_detection_history)} frames")
                
                # Trigger lost transition if in visual servoing
                if self.state == "visual_servoing":
                    self.visual_target_lost()
        else:
            # Not enough history yet
            self.aruco_detected_consistent = False
            rospy.loginfo_throttle(1.0, f"Building detection history: {len(self.aruco_detection_history)}/{self.aruco_detection_window_size} frames")
        
    def task_done_callback(self, msg):
        if msg.data and self.state == "performing_task":
            self.task_completed()
            
    def mavros_state_callback(self, msg):
        self.mavros_state = msg
        
    def pose_callback(self, msg):
        self.current_pose = msg
        # Update distance whenever pose updates
        self.update_target_distance()

    def check_goal_status(self, event=None):
        # Publish active goal continuously
        self.publish_active_goal()
        
        if self.state == "visual_servoing":
            if self.check_visual_goal_reached_conditions():
                rospy.loginfo("Visual goal reached, triggering transition to performing_task")
                self.visual_goal_reached()
        elif self.state == "gps_navigation":
            if self.check_gps_goal_reached_conditions():
                rospy.loginfo("GPS goal reached, triggering transition to search_for_object or next GPS goal")
                self.gps_goal_reached()

    # Actions
    def increment_goal_index(self, event=None):
        self.current_mission_goal_index += 1
        rospy.loginfo(f"Moving to goal {self.current_mission_goal_index + 1}/{len(self.goal_points)}")
        
    def skip_to_next_goal(self, event=None):
        """Skip current goal and move to next one (used when search times out)"""
        rospy.logwarn(f"Skipping goal {self.current_mission_goal_index + 1} - target not found")
        self.increment_goal_index()
        
        # Check if we have more goals
        if self.current_mission_goal_index >= len(self.goal_points):
            rospy.loginfo("No more goals, returning home")
            self.trigger('mission_complete')  # This will transition to returning_home
        
    def set_navigation_mode(self, mode):
        """Directly set navigation mode and publish it"""
        self.navigation_mode = mode
        self.mode_pub.publish(String(mode))
        rospy.loginfo(f"Navigation mode set to: {mode}")
        
        # Select and publish appropriate goal based on mode
        self.publish_active_goal()
        return True
    
    def publish_active_goal(self):
        """Publish the active goal based on current navigation mode"""
        if self.navigation_mode == "gps_tracking" and self.gps_target:
            self.gps_goal_pub.publish(self.gps_target)
            self.active_target = self.gps_target
        elif self.navigation_mode == "visual_servoing" and self.visual_target:
            self.visual_goal_pub.publish(self.visual_target)
            self.active_target = self.visual_target
        elif self.navigation_mode == "search":
            # In search mode, might hover at current GPS target or implement search pattern
            if self.gps_target:
                self.gps_goal_pub.publish(self.gps_target)
                self.active_target = self.gps_target
        else:
            self.active_target = None
            
    def arm_and_set_offboard(self):
        """Arm the drone and set OFFBOARD mode"""
        # Arm
        try:
            arm_resp = self.arming_client(True)
            if not arm_resp.success:
                rospy.logwarn("Failed to arm drone")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service failed: {e}")
            return False
            
        # Set OFFBOARD
        try:
            mode_req = SetModeRequest()
            mode_req.custom_mode = "OFFBOARD"
            mode_resp = self.mavros_mode_client.call(mode_req)
            if not mode_resp.mode_sent:
                rospy.logwarn("Failed to set OFFBOARD mode")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Set mode service failed: {e}")
            return False
            
        return True
        
    def set_hover_mode(self):
        """Set drone to hover mode for safety during transitions"""
        try:
            mode_req = SetModeRequest()
            mode_req.custom_mode = "AUTO.LOITER"
            mode_resp = self.mavros_mode_client.call(mode_req)
            if mode_resp.mode_sent:
                rospy.loginfo("Hover mode (AUTO.LOITER) set for safety")
            else:
                rospy.logwarn("Failed to set hover mode")
        except rospy.ServiceException as e:
            rospy.logerr(f"Hover mode service failed: {e}")
        
    def publish_mission_goal(self):
        """Publish current mission goal"""
        if self.current_mission_goal_index < len(self.goal_points):
            goal = PoseStamped()
            goal.header.frame_id = "odom"
            goal.header.stamp = rospy.Time.now()
            
            coords = self.goal_points[self.current_mission_goal_index]
            goal.pose.position.x = coords[0]
            goal.pose.position.y = coords[1]
            goal.pose.position.z = coords[2]
            goal.pose.orientation.w = 1.0
            
            # Store as GPS target and publish to mission topic
            self.gps_target = goal
            self.mission_goal_pub.publish(goal)
            rospy.loginfo(f"Published mission goal: {coords}")
            
    # State entry functions
    def on_enter_idle(self, event=None):
        rospy.loginfo("STATE: IDLE - Waiting for mission start")
        self.set_navigation_mode("hold")
        self.state_pub.publish(String("idle"))
        
    def on_enter_gps_navigation(self, event=None):
        rospy.loginfo(f"STATE: GPS_NAVIGATION - Goal index: {self.current_mission_goal_index}/{len(self.goal_points)}")
        self.arm_and_set_offboard()
        self.set_navigation_mode("gps_tracking")
        self.publish_mission_goal()
        self.state_pub.publish(String("gps_navigation"))
        
    def on_exit_gps_navigation(self, event=None):
        """Safety: Set hover mode when leaving GPS navigation"""
        # self.set_hover_mode()
        rospy.loginfo("Exiting GPS_NAVIGATION")
        
    def on_enter_visual_servoing(self, event=None):
        rospy.loginfo("STATE: VISUAL_SERVOING - Switching to visual control")
        rospy.loginfo(f"Transition conditions met - ArUco: {self.aruco_detected_consistent}, Distance: {self.target_distance:.2f}m, Near GPS: {self.is_near_gps_goal()}")
        self.arm_and_set_offboard()  # Ensure OFFBOARD mode
        self.set_navigation_mode("visual_servoing")
        self.state_pub.publish(String("visual_servoing"))
        
    def on_exit_visual_servoing(self, event=None):
        """Safety: Set hover mode when leaving visual servoing"""
        self.set_hover_mode()
        rospy.loginfo("Exiting VISUAL_SERVOING - hover mode set")
        
    def on_enter_search_for_object(self, event=None):
        rospy.loginfo("STATE: SEARCH_FOR_OBJECT - Searching for ArUco marker")
        self.arm_and_set_offboard()  # Ensure OFFBOARD mode
        self.set_navigation_mode("search")
        
        # Start search timeout
        if self.search_timer:
            self.search_timer.cancel()
        self.search_timer = Timer(self.search_timeout, self.search_timeout_handler)
        self.search_timer.start()
        
        self.state_pub.publish(String("search_for_object"))
        
    def on_exit_search_for_object(self, event=None):
        """Safety: Cancel timer and set hover mode"""
        if self.search_timer:
            self.search_timer.cancel()
            self.search_timer = None
        self.set_hover_mode()
        rospy.loginfo("Exiting SEARCH_FOR_OBJECT - hover mode set")
        
    def search_timeout_handler(self):
        rospy.logwarn("Search timeout - returning to GPS navigation")
        self.search_timeout_triggered()
        
    def on_enter_performing_task(self, event=None):
        rospy.loginfo("STATE: PERFORMING_TASK - Executing injection")
        # Here you would trigger the injection mechanism
        # For now, we'll simulate with a timer
        self.set_navigation_mode("hold")
        rospy.Timer(rospy.Duration(5.0), lambda e: self.task_done_callback(Bool(True)), oneshot=True)
        self.state_pub.publish(String("performing_task"))
        
    def on_exit_performing_task(self, event=None):
        """Safety: Set hover mode after task"""
        self.set_hover_mode()
        rospy.loginfo("Exiting PERFORMING_TASK - hover mode set")
        
    def on_enter_returning_home(self, event=None):
        rospy.loginfo("STATE: RETURNING_HOME")
        self.arm_and_set_offboard()  # Ensure OFFBOARD mode
        self.set_navigation_mode("gps_tracking")
        
        # Publish home position as goal
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.home_position[0]
        goal.pose.position.y = self.home_position[1]
        goal.pose.position.z = self.home_position[2]
        goal.pose.orientation.w = 1.0
        
        # Store as GPS target and publish
        self.gps_target = goal
        self.mission_goal_pub.publish(goal)
        self.state_pub.publish(String("returning_home"))
        
    def on_exit_returning_home(self, event=None):
        """Safety: Set hover mode before landing"""
        self.set_hover_mode()
        rospy.loginfo("Exiting RETURNING_HOME - hover mode set")
        
    def on_enter_landing(self, event=None):
        rospy.loginfo("STATE: LANDING - Mission complete")
        self.set_navigation_mode("hold")
        
        # Set AUTO.LAND mode
        try:
            mode_req = SetModeRequest()
            mode_req.custom_mode = "AUTO.LAND"
            self.mavros_mode_client.call(mode_req)
        except:
            rospy.logwarn("Failed to set landing mode")
            
        self.state_pub.publish(String("landing"))
        
    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)
        
        # Wait for systems to initialize
        rospy.sleep(2.0)
        
        # Start mission if we have goals
        if self.has_mission_goals():
            self.start_mission()
            
        while not rospy.is_shutdown():
            # State machine runs on callbacks
            rate.sleep()
            
if __name__ == '__main__':
    rospy.init_node('enhanced_drone_state_machine')
    
    try:
        sm = EnhancedDroneStateMachine()
        sm.run()
    except rospy.ROSInterruptException:
        pass