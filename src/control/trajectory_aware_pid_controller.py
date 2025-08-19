#!/usr/bin/env python3
import rospy
import numpy as np
import math
from enum import Enum
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_msgs.msg import String, Float32
from tf.transformations import euler_from_quaternion

class YawControlMode(Enum):
    VELOCITY_ALIGNED = "velocity_aligned"  # Face direction of motion
    TARGET_LOCKED = "target_locked"        # Face the target
    HYBRID = "hybrid"                       # Smart switching based on distance

class ControlMode(Enum):
    POSITION_SETPOINT = "position_setpoint"  # Single goal position
    TRAJECTORY_TRACKING = "trajectory_tracking"  # Follow a path
    VELOCITY_DIRECT = "velocity_direct"  # Direct velocity commands

class InputPriority(Enum):
    EMERGENCY = 0  # Highest priority - safety override
    VISUAL = 1     # Visual servoing for injection
    TRAJECTORY = 2 # Global planner trajectory
    GPS = 3        # GPS waypoint - lowest priority

class TrajectoryAwarePIDController:
    """
    Enhanced PID controller that can handle:
    1. Position setpoints (for GPS waypoints and visual servoing)
    2. Trajectory tracking (for global planner paths)
    3. Direct velocity commands (for emergency maneuvers)
    """
    
    def __init__(self):
        rospy.init_node('trajectory_aware_pid_controller')
        
        # Control mode and priority
        self.control_mode = ControlMode.POSITION_SETPOINT
        self.current_priority = InputPriority.GPS
        self.last_input_time = {}  # Track when each input was last received
        self.input_timeout = 2.0  # seconds before an input is considered stale
        
        # Load parameters
        self.load_parameters()
        
        # State variables
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_yaw = 0.0
        self.navigation_mode = "gps_tracking"  # Initialize navigation mode
        self.target_distance = float('inf')
        self.yaw_control_mode = None
        
        # Position setpoint mode variables
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.gps_target = None  # Store GPS target separately
        self.visual_target = None  # Store visual target separately
        
        # Trajectory tracking variables
        self.trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0
        self.carrot_distance = 2.0  # Look-ahead distance
        self.trajectory_poses = []  # Store as list of poses
        
        # PID state
        self.integral = np.zeros(4)
        self.previous_error = np.zeros(4)
        self.last_time = rospy.Time.now()
        
        # Setup ROS interfaces
        self.setup_publishers()
        self.setup_subscribers()
        
        self.rate = rospy.Rate(self.control_frequency)
        rospy.loginfo("Trajectory-Aware PID Controller initialized")
        
    def load_parameters(self):
        # Priority and conflict resolution
        self.priority_mode = rospy.get_param('~priority_mode', 'auto')  # 'auto', 'manual', 'navigation_mode'
        self.input_timeout = rospy.get_param('~input_timeout', 2.0)  # seconds
        self.allow_override = rospy.get_param('~allow_override', True)

        # Loop Rate
        self.control_frequency = rospy.get_param('~control_frequency', 20.0)
        
        # Control modes parameters
        self.enable_trajectory_mode = rospy.get_param('~enable_trajectory_mode', True)
        self.trajectory_lookahead = rospy.get_param('~trajectory_lookahead', 3.0)  # meters
        self.trajectory_convergence_radius = rospy.get_param('~trajectory_convergence_radius', 1.0)
        
        # PID gains (same structure as before, but we'll use them differently)
        self.kp_position = np.array(rospy.get_param('~kp_position', [0.8, 0.8, 0.6, 1.0]))
        self.ki_position = np.array(rospy.get_param('~ki_position', [0.02, 0.02, 0.02, 0.0]))
        self.kd_position = np.array(rospy.get_param('~kd_position', [0.15, 0.15, 0.12, 0.15]))
        
        # Trajectory tracking gains (typically different from position control)
        self.kp_trajectory = np.array(rospy.get_param('~kp_trajectory', [1.2, 1.2, 0.8, 1.0]))
        self.ki_trajectory = np.array(rospy.get_param('~ki_trajectory', [0.0, 0.0, 0.0, 0.0])) # New Integral gains
        self.kd_trajectory = np.array(rospy.get_param('~kd_trajectory', [0.3, 0.3, 0.2, 0.2]))
        self.trajectory_feedforward_gain = rospy.get_param('~trajectory_feedforward_gain', 0.8)
        
        # Velocity limits
        self.max_velocity = rospy.get_param('~max_velocity', 2.0)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 0.5)
        
        # Goal tolerances
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.5)
        self.trajectory_tolerance = rospy.get_param('~trajectory_tolerance', 1.0)

        # Hybrid Yaw Parameters (moved from navigation_mode_manager)
        self.yaw_distance_threshold = rospy.get_param('~yaw_distance_threshold', 60.0)
        self.yaw_injection_distance = rospy.get_param('~yaw_injection_distance', 3.0)

        # Search Parameters
        self.search_yaw_rate = rospy.get_param('~search_yaw_rate', 0.2)
        
    def setup_publishers(self):
        self.vel_pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=10)
        self.control_mode_pub = rospy.Publisher('/pid/control_mode', String, queue_size=1)
        self.priority_pub = rospy.Publisher('/pid/current_priority', String, queue_size=1)
        self.conflict_pub = rospy.Publisher('/pid/input_conflict', String, queue_size=1)
        self.carrot_pub = rospy.Publisher('/pid/carrot_point', PoseStamped, queue_size=1)  # For visualization
        self.tracking_error_pub = rospy.Publisher('/pid/tracking_error', Float32, queue_size=1)
        
    def setup_subscribers(self):
        # Odometry
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        
        # Position setpoint inputs with different priorities
        rospy.Subscriber('/move_base_gps', PoseStamped, self.gps_goal_callback)
        rospy.Subscriber('/move_base_visual', PoseStamped, self.visual_goal_callback)
        
        # Trajectory inputs (new)
        rospy.Subscriber('/global_planner/trajectory', Path, self.trajectory_callback)
        rospy.Subscriber('/global_planner/multi_dof_trajectory', 
                        MultiDOFJointTrajectory, self.multi_dof_trajectory_callback)
        
        # Direct velocity commands (for override)
        rospy.Subscriber('/emergency/velocity_command', TwistStamped, self.direct_velocity_callback)
        
        # Navigation mode from mode manager
        rospy.Subscriber('/navigation/current_mode', String, self.navigation_mode_callback)
        rospy.Subscriber('/navigation/target_distance', Float32, self.distance_callback) # For hybrid yaw
        
    def odom_callback(self, msg):
        self.current_pos[0] = msg.pose.pose.position.x
        self.current_pos[1] = msg.pose.pose.position.y
        self.current_pos[2] = msg.pose.pose.position.z
        
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.linear.z
        
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
    def navigation_mode_callback(self, msg):
        """Track navigation mode from mode manager"""
        self.navigation_mode = msg.data
        # Add logic here to reset priority
        if self.navigation_mode == "gps_tracking":
            # When returning to GPS tracking, reset priority to GPS
            # This allows GPS goals to be accepted again
            if self.current_priority.value < InputPriority.GPS.value: # Only reset if current is higher priority
                rospy.loginfo(f"Navigation mode changed to GPS_TRACKING. Resetting PID priority from {self.current_priority.name} to GPS.")
                self.current_priority = InputPriority.GPS
                self.control_mode = ControlMode.POSITION_SETPOINT # Ensure control mode is appropriate
                self.last_input_time[InputPriority.GPS] = rospy.Time.now() # Reset timestamp to prevent immediate staleness
                # Explicitly re-process the last GPS target to ensure it's accepted with the new priority
                if self.gps_target:
                    rospy.loginfo("Re-processing last GPS target after priority reset.")
                    self.gps_goal_callback(self.gps_target)

    def distance_callback(self, msg):
        self.target_distance = msg.data

    def calculate_hybrid_yaw(self):
        # Implements the hybrid yaw control strategy
        if not self.gps_target:
            return 0.0

        desired_yaw = self.target_yaw # Default to the final desired yaw

        if self.target_distance > self.yaw_distance_threshold:
            # Far away - align with velocity
            if np.linalg.norm(self.current_vel[:2]) > 0.1:
                desired_yaw = math.atan2(self.current_vel[1], self.current_vel[0])
                self.yaw_control_mode = YawControlMode.VELOCITY_ALIGNED
        elif self.target_distance < self.yaw_injection_distance:
            # Very close - use the final orientation from the goal
            desired_yaw = self.target_yaw
            self.yaw_control_mode = YawControlMode.TARGET_LOCKED
        else:
            # Medium distance - face toward the target
            pos_error = self.target_pos - self.current_pos
            desired_yaw = math.atan2(pos_error[1], pos_error[0])
            self.yaw_control_mode = YawControlMode.TARGET_LOCKED

        # Calculate yaw error
        yaw_error = desired_yaw - self.current_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Apply P-control to get yaw rate
        # yaw_rate = self.yaw_p_gain * yaw_error
        return desired_yaw
        
    def should_accept_input(self, new_priority):
        """Determine if new input should override current control"""
        current_time = rospy.Time.now()
        
        # Always accept emergency commands
        if new_priority == InputPriority.EMERGENCY:
            return True
            
        # Check if current input is stale
        if self.current_priority in self.last_input_time:
            time_since_last = (current_time - self.last_input_time[self.current_priority]).to_sec()
            if time_since_last > self.input_timeout:
                rospy.loginfo(f"Current input ({self.current_priority.name}) is stale, accepting new input")
                return True
                
        # Priority-based override
        if self.allow_override and new_priority.value < self.current_priority.value:
            rospy.loginfo(f"Higher priority input ({new_priority.name}) overriding {self.current_priority.name}")
            return True
            
        # If same priority, accept (update target)
        if new_priority == self.current_priority:
            return True
            
        # Otherwise, reject
        # rospy.logwarn(f"Rejecting {new_priority.name} input - current priority is {self.current_priority.name}")
        return False
        
    def gps_goal_callback(self, msg):
        """Handle GPS waypoint with lowest priority"""
        if not self.should_accept_input(InputPriority.GPS):
            return
            
        self.control_mode = ControlMode.POSITION_SETPOINT
        self.current_priority = InputPriority.GPS
        self.last_input_time[InputPriority.GPS] = rospy.Time.now()
        
        self.gps_target = msg
        self.target_pos[0] = msg.pose.position.x
        self.target_pos[1] = msg.pose.position.y
        self.target_pos[2] = msg.pose.position.z
        
        q = msg.pose.orientation
        _, _, self.target_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Reset integrals for new setpoint
        self.integral = np.zeros(4)
        
        # rospy.loginfo(f"GPS position mode: target=({self.target_pos[0]:.2f}, "
        #              f"{self.target_pos[1]:.2f}, {self.target_pos[2]:.2f})")
                     
    def visual_goal_callback(self, msg):
        """Handle visual servoing target with high priority"""
        # Only accept visual input if in visual servoing mode
        if self.navigation_mode != "visual_servoing":
            return

        if not self.should_accept_input(InputPriority.VISUAL):
            return
            
        self.control_mode = ControlMode.POSITION_SETPOINT
        self.current_priority = InputPriority.VISUAL
        self.last_input_time[InputPriority.VISUAL] = rospy.Time.now()
        
        self.visual_target = msg
        self.target_pos[0] = msg.pose.position.x
        self.target_pos[1] = msg.pose.position.y
        self.target_pos[2] = msg.pose.position.z
        
        q = msg.pose.orientation
        _, _, self.target_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Reset integrals for new setpoint
        self.integral = np.zeros(4)
        
        # # rospy.loginfo(f"VISUAL servoing mode: target=({self.target_pos[0]:.2f}, "
        # #              f"{self.target_pos[1]:.2f}, {self.target_pos[2]:.2f})")
        
    def trajectory_callback(self, msg):
        """Handle trajectory from global planner with medium priority"""
        if not self.should_accept_input(InputPriority.TRAJECTORY):
            return
            
        if not self.enable_trajectory_mode:
            # Convert to waypoint following
            self.convert_trajectory_to_waypoints(msg)
            return
            
        self.control_mode = ControlMode.TRAJECTORY_TRACKING
        self.current_priority = InputPriority.TRAJECTORY
        self.last_input_time[InputPriority.TRAJECTORY] = rospy.Time.now()
        
        self.trajectory_poses = msg.poses
        self.trajectory_index = 0
        self.trajectory_start_time = rospy.Time.now()
        
        # Reset integrals for new trajectory
        self.integral = np.zeros(4)
        
        rospy.loginfo(f"TRAJECTORY tracking mode: {len(self.trajectory_poses)} points")
        
    def multi_dof_trajectory_callback(self, msg):
        """Handle multi-DOF trajectory with velocity/acceleration info"""
        self.control_mode = ControlMode.TRAJECTORY_TRACKING
        self.trajectory = msg
        self.trajectory_index = 0
        self.trajectory_start_time = rospy.Time.now()
        
        rospy.loginfo(f"Multi-DOF trajectory mode: {len(msg.points)} points with feedforward")
        
    def direct_velocity_callback(self, msg):
        """Emergency velocity override"""
        self.control_mode = ControlMode.VELOCITY_DIRECT
        self.vel_pub.publish(msg)
        rospy.logwarn("Direct velocity override active!")
        
    def convert_trajectory_to_waypoints(self, path_msg):
        """Fallback: treat trajectory as sequence of waypoints"""
        # Simple approach: just take next waypoint
        if self.trajectory_index < len(path_msg.poses):
            self.position_goal_callback(path_msg.poses[self.trajectory_index])
            # Check if close enough to advance
            dist = np.linalg.norm(self.target_pos - self.current_pos)
            if dist < self.trajectory_tolerance:
                self.trajectory_index += 1
                
    def find_carrot_point(self):
        """Find look-ahead point on trajectory for smooth following"""
        if not self.trajectory_poses:
            return None
            
        # Find closest point on trajectory
        min_dist = float('inf')
        closest_idx = self.trajectory_index
        
        for i in range(self.trajectory_index, min(len(self.trajectory_poses), 
                                                   self.trajectory_index + 20)):
            pose = self.trajectory_poses[i].pose
            dist = np.linalg.norm([
                pose.position.x - self.current_pos[0],
                pose.position.y - self.current_pos[1],
                pose.position.z - self.current_pos[2]
            ])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Find carrot point at look-ahead distance
        accumulated_dist = 0.0
        carrot_idx = closest_idx
        
        for i in range(closest_idx, len(self.trajectory_poses) - 1):
            pose1 = self.trajectory_poses[i].pose
            pose2 = self.trajectory_poses[i + 1].pose
            
            segment_dist = np.linalg.norm([
                pose2.position.x - pose1.position.x,
                pose2.position.y - pose1.position.y,
                pose2.position.z - pose1.position.z
            ])
            
            if accumulated_dist + segment_dist >= self.carrot_distance:
                # Interpolate carrot point
                ratio = (self.carrot_distance - accumulated_dist) / segment_dist
                carrot_point = PoseStamped()
                carrot_point.header.frame_id = "odom"
                carrot_point.header.stamp = rospy.Time.now()
                
                carrot_point.pose.position.x = (pose1.position.x + 
                                               ratio * (pose2.position.x - pose1.position.x))
                carrot_point.pose.position.y = (pose1.position.y + 
                                               ratio * (pose2.position.y - pose1.position.y))
                carrot_point.pose.position.z = (pose1.position.z + 
                                               ratio * (pose2.position.z - pose1.position.z))
                carrot_point.pose.orientation = pose2.orientation  # Use next orientation
                
                self.trajectory_index = i  # Update progress
                return carrot_point
                
            accumulated_dist += segment_dist
            
        # Return last point if we're near the end
        return self.trajectory_poses[-1] if self.trajectory_poses else None
        
    def compute_trajectory_tracking_control(self, dt):
        """Compute control for trajectory tracking mode"""
        carrot_point = self.find_carrot_point()
        
        if not carrot_point:
            rospy.logwarn("No carrot point found")
            return self.compute_position_control(dt)  # Fallback
            
        # Publish carrot for visualization
        self.carrot_pub.publish(carrot_point)
        
        # Position error to carrot point
        carrot_pos = np.array([
            carrot_point.pose.position.x,
            carrot_point.pose.position.y,
            carrot_point.pose.position.z
        ])
        
        pos_error = carrot_pos - self.current_pos
        
        # Get desired yaw from carrot orientation
        q = carrot_point.pose.orientation
        _, _, desired_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_error = desired_yaw - self.current_yaw
        
        # Normalize yaw error
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
            
        # Tracking error for monitoring
        tracking_error = np.linalg.norm(pos_error)
        self.tracking_error_pub.publish(Float32(tracking_error))
        
        # Full PID control for trajectory
        error = np.array([pos_error[0], pos_error[1], pos_error[2], yaw_error])
        
        # Calculate derivative and integral
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)  # Anti-windup

        if np.any(self.previous_error):
            derivative = (error - self.previous_error) / dt
        else:
            derivative = np.zeros(4)
            
        # PID control with trajectory-specific gains
        output = (self.kp_trajectory * error + 
                  self.ki_trajectory * self.integral + 
                  self.kd_trajectory * derivative)
        
        # Add feedforward if we have velocity information
        if hasattr(self, 'trajectory') and self.trajectory:
            # Get feedforward velocities from trajectory
            if self.trajectory_index < len(self.trajectory.points):
                point = self.trajectory.points[self.trajectory_index]
                ff_vel = np.array([
                    point.velocities[0].linear.x,
                    point.velocities[0].linear.y,
                    point.velocities[0].linear.z
                ])
                output[:3] += self.trajectory_feedforward_gain * ff_vel
                
        self.previous_error = error
        
        return output
        
    def compute_position_control(self, dt):
        """Compute control for position setpoint mode (existing PID)"""
        pos_error = self.target_pos - self.current_pos
        
        # Get desired yaw from our hybrid logic
        desired_yaw = self.calculate_hybrid_yaw()
        yaw_error = desired_yaw - self.current_yaw
        
        # Normalize yaw error
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
            
        # Full PID control
        error = np.array([pos_error[0], pos_error[1], pos_error[2], yaw_error])
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)  # Anti-windup
        
        if np.any(self.previous_error):
            derivative = (error - self.previous_error) / dt
        else:
            derivative = np.zeros(4)
            
        output = (self.kp_position * error + 
                 self.ki_position * self.integral + 
                 self.kd_position * derivative)
        
        self.previous_error = error
        
        return output
        
    def run(self):
        while not rospy.is_shutdown():
            # Check for highest-priority override first
            if self.control_mode == ControlMode.VELOCITY_DIRECT:
                # Emergency command is published directly from its callback.
                # This loop should wait to prevent publishing conflicting commands.
                self.rate.sleep()
                continue

            # If in search mode, override all other logic and just rotate.
            if self.navigation_mode == 'search':
                vel_msg = TwistStamped()
                vel_msg.header.stamp = rospy.Time.now()
                vel_msg.header.frame_id = "odom"
                vel_msg.twist.linear.x = 0
                vel_msg.twist.linear.y = 0
                vel_msg.twist.linear.z = 0
                vel_msg.twist.angular.z = self.search_yaw_rate
                self.vel_pub.publish(vel_msg)
                self.rate.sleep()
                continue

            # Calculate dt dynamically for PID controllers
            current_time = rospy.Time.now()
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time

            if dt <= 0:
                self.rate.sleep()
                continue

            # Compute control based on mode
            if self.control_mode == ControlMode.POSITION_SETPOINT:
                output = self.compute_position_control(dt)
            elif self.control_mode == ControlMode.TRAJECTORY_TRACKING:
                output = self.compute_trajectory_tracking_control(dt)
            elif self.control_mode == ControlMode.VELOCITY_DIRECT:
                # Direct velocity mode handled in callback
                self.rate.sleep()
                continue
            else:
                output = np.zeros(4)
                
            # Extract linear and angular components
            linear_vel = output[:3]
            yaw_rate = output[3]
            
            # Apply velocity limits
            vel_norm = np.linalg.norm(linear_vel)
            if vel_norm > self.max_velocity:
                linear_vel = (linear_vel / vel_norm) * self.max_velocity
                
            # yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
            
            # # DEBUGGING: Log the output before publishing
            # # rospy.loginfo(f"[PID DEBUG] Output Vel: linear=[{linear_vel[0]:.2f}, {linear_vel[1]:.2f}, {linear_vel[2]:.2f}], angular_z=[{yaw_rate:.2f}]")

            # Publish velocity command
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.header.frame_id = "odom"
            vel_msg.twist.linear.x = linear_vel[0]
            vel_msg.twist.linear.y = linear_vel[1]
            vel_msg.twist.linear.z = linear_vel[2]
            vel_msg.twist.angular.z = yaw_rate
            
            self.vel_pub.publish(vel_msg)
            
            # Publish mode for monitoring
            self.control_mode_pub.publish(String(self.control_mode.value))
            
            self.rate.sleep()
            
if __name__ == '__main__':
    try:
        controller = TrajectoryAwarePIDController()
        controller.run()
    except rospy.ROSInterruptException:
        pass