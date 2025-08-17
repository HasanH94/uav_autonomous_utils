#!/usr/bin/env python3
import rospy
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State
from std_msgs.msg import Bool, String, Float32
from tf.transformations import euler_from_quaternion

class EnhancedGoalToVelocityPID:
    def __init__(self):
        rospy.init_node('enhanced_goal_to_velocity_pid')

        # Navigation modes
        self.NAVIGATION_MODE_GPS = "gps_tracking"
        self.NAVIGATION_MODE_VISUAL = "visual_servoing"
        self.navigation_mode = self.NAVIGATION_MODE_GPS
        
        # State Machine for local control
        self.STATE_NAVIGATING = "NAVIGATING"
        self.STATE_APPROACHING = "APPROACHING"  # Slower, more precise
        self.STATE_HOLDING_POSITION = "HOLDING_POSITION"
        self.current_state = self.STATE_NAVIGATING

        # Load parameters
        self.load_parameters()
        
        # PID state variables
        self.integral = np.zeros(4)
        self.previous_error = np.zeros(4)
        self.last_time = rospy.Time.now()
        
        # Tracking variables
        self.current_pos = np.zeros(3)
        self.current_yaw = 0.0
        self.current_vel = np.zeros(3)
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.target_yaw = 0.0
        self.target_distance = float('inf')
        self.mavros_current_state = State()
        
        # Yaw control state
        self.yaw_mode = "velocity_aligned"  # or "target_locked"
        self.computed_yaw_rate = 0.0
        
        # Adaptive gain state
        self.current_kp = self.kp_cruise.copy()
        self.current_kd = self.kd_cruise.copy()
        
        # Publishers
        self.vel_pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=10)
        self.goal_reached_pub = rospy.Publisher('/drone_events/reached_goal_point', Bool, queue_size=1)
        self.pid_state_pub = rospy.Publisher('/pid/state', String, queue_size=1)
        self.pid_gains_pub = rospy.Publisher('/pid/current_gains', String, queue_size=1)
        
        # Services
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        
        # Goal inputs - now we have separate topics for different modes
        rospy.Subscriber('/move_base_gps', PoseStamped, self.gps_goal_callback)
        rospy.Subscriber('/move_base_visual', PoseStamped, self.visual_goal_callback)
        
        # Navigation mode updates
        rospy.Subscriber('/navigation/current_mode', String, self.navigation_mode_callback)
        rospy.Subscriber('/navigation/target_distance', Float32, self.target_distance_callback)
        rospy.Subscriber('/navigation/yaw_mode', String, self.yaw_mode_callback)
        
        self.rate = rospy.Rate(20.0)
        rospy.on_shutdown(self.shutdown_cb)
        
        rospy.loginfo("Enhanced Goal-to-Velocity PID controller initialized")
        
    def load_parameters(self):
        # Distance thresholds
        self.yaw_distance_threshold = rospy.get_param('~yaw_distance_threshold', 60.0)
        self.approach_distance = rospy.get_param('~approach_distance', 10.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)
        self.visual_goal_tolerance = rospy.get_param('~visual_goal_tolerance', 0.2)  # Tighter for injection
        
        # Velocity limits
        self.max_speed_cruise = rospy.get_param('~max_speed_cruise', 3.0)
        self.max_speed_approach = rospy.get_param('~max_speed_approach', 1.0)
        self.max_speed_visual = rospy.get_param('~max_speed_visual', 0.5)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 0.5)
        
        # PID Gains for different phases
        # Cruise gains (long distance)
        self.kp_cruise = np.array(rospy.get_param('~kp_cruise', [0.5, 0.5, 0.5, 0.8]))
        self.ki_cruise = np.array(rospy.get_param('~ki_cruise', [0.01, 0.01, 0.01, 0.0]))
        self.kd_cruise = np.array(rospy.get_param('~kd_cruise', [0.1, 0.1, 0.1, 0.1]))
        
        # Approach gains (medium distance)
        self.kp_approach = np.array(rospy.get_param('~kp_approach', [0.8, 0.8, 0.6, 1.0]))
        self.ki_approach = np.array(rospy.get_param('~ki_approach', [0.02, 0.02, 0.02, 0.0]))
        self.kd_approach = np.array(rospy.get_param('~kd_approach', [0.15, 0.15, 0.12, 0.15]))
        
        # Visual servoing gains (close distance, high precision)
        self.kp_visual = np.array(rospy.get_param('~kp_visual', [1.2, 1.2, 0.8, 1.2]))
        self.ki_visual = np.array(rospy.get_param('~ki_visual', [0.05, 0.05, 0.03, 0.0]))
        self.kd_visual = np.array(rospy.get_param('~kd_visual', [0.2, 0.2, 0.15, 0.2]))
        
        # Velocity profiling parameters
        self.acceleration_limit = rospy.get_param('~acceleration_limit', 1.0)  # m/s^2
        self.deceleration_distance = rospy.get_param('~deceleration_distance', 5.0)
        
        # Other parameters
        self.hover_mode = rospy.get_param('~hover_mode', "AUTO.LOITER")
        self.enable_adaptive_gains = rospy.get_param('~enable_adaptive_gains', True)
        self.enable_velocity_profiling = rospy.get_param('~enable_velocity_profiling', True)
        
    def odom_callback(self, msg):
        self.current_pos[0] = msg.pose.pose.position.x
        self.current_pos[1] = msg.pose.pose.position.y
        self.current_pos[2] = msg.pose.pose.position.z
        
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.linear.z
        
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
    def mavros_state_callback(self, msg):
        self.mavros_current_state = msg
        
    def navigation_mode_callback(self, msg):
        self.navigation_mode = msg.data
        rospy.loginfo(f"Navigation mode changed to: {self.navigation_mode}")
        
    def target_distance_callback(self, msg):
        self.target_distance = msg.data
        
    def yaw_mode_callback(self, msg):
        self.yaw_mode = msg.data
        
    def gps_goal_callback(self, msg):
        if self.navigation_mode == self.NAVIGATION_MODE_GPS:
            self.update_target(msg)
            
    def visual_goal_callback(self, msg):
        if self.navigation_mode == self.NAVIGATION_MODE_VISUAL:
            self.update_target(msg)
            
    def update_target(self, msg):
        new_target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # Extract target yaw from quaternion
        q = msg.pose.orientation
        _, _, new_target_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Check if this is a significant change
        if np.linalg.norm(new_target_pos - self.target_pos) > 0.05:
            self.target_pos = new_target_pos
            self.target_yaw = new_target_yaw
            
            # Reset PID integrals for new target
            self.integral = np.zeros(4)
            self.previous_error = np.zeros(4)
            
            # Switch to navigating state
            if self.current_state == self.STATE_HOLDING_POSITION:
                self.switch_state(self.STATE_NAVIGATING)
                self.set_offboard_mode()
                
            rospy.loginfo(f"New target: Pos=({new_target_pos[0]:.2f}, {new_target_pos[1]:.2f}, {new_target_pos[2]:.2f})")
            
    def switch_state(self, new_state):
        if self.current_state != new_state:
            rospy.loginfo(f"PID State: {self.current_state} -> {new_state}")
            self.current_state = new_state
            self.pid_state_pub.publish(String(new_state))
            
    def update_adaptive_gains(self):
        """Adapt PID gains based on distance and navigation mode"""
        if not self.enable_adaptive_gains:
            return
            
        # Select base gains based on mode and distance
        if self.navigation_mode == self.NAVIGATION_MODE_VISUAL:
            self.current_kp = self.kp_visual
            self.current_kd = self.kd_visual
            base_ki = self.ki_visual
            max_speed = self.max_speed_visual
        elif self.target_distance < self.approach_distance:
            self.current_kp = self.kp_approach
            self.current_kd = self.kd_approach
            base_ki = self.ki_approach
            max_speed = self.max_speed_approach
            self.switch_state(self.STATE_APPROACHING)
        else:
            self.current_kp = self.kp_cruise
            self.current_kd = self.kd_cruise
            base_ki = self.ki_cruise
            max_speed = self.max_speed_cruise
            if self.current_state == self.STATE_APPROACHING:
                self.switch_state(self.STATE_NAVIGATING)
                
        # Smooth gain transitions
        alpha = 0.1  # Smoothing factor
        self.integral = self.integral * (1 - alpha) + np.zeros(4) * alpha  # Decay integral during transitions
        
        # Publish current gains for debugging
        gains_str = f"kp:{self.current_kp[0]:.2f}, kd:{self.current_kd[0]:.2f}, max_v:{max_speed:.1f}"
        self.pid_gains_pub.publish(String(gains_str))
        
        return max_speed
        
    def calculate_velocity_profile(self, distance, current_speed):
        """Calculate desired velocity based on distance (trapezoidal profile)"""
        if not self.enable_velocity_profiling:
            return self.max_speed_cruise
            
        # Deceleration phase
        if distance < self.deceleration_distance:
            # v = sqrt(2 * a * d) for constant deceleration
            desired_speed = math.sqrt(2 * self.acceleration_limit * distance)
            return min(desired_speed, current_speed + self.acceleration_limit * 0.05)  # Rate limit
        else:
            # Cruise or acceleration phase
            return self.max_speed_cruise
            
    def compute_hybrid_yaw_rate(self, pos_error):
        """
        Compute yaw rate based on hybrid strategy:
        - Far away (>60m): align with velocity
        - Medium (3-60m): face toward the target
        - Close (<3m) in visual mode: use exact target orientation
        """
        if self.yaw_mode == "velocity_aligned" and self.target_distance > self.yaw_distance_threshold:
            # Align with velocity direction
            if np.linalg.norm(self.current_vel[:2]) > 0.2:  # Moving
                desired_yaw = math.atan2(self.current_vel[1], self.current_vel[0])
            else:
                # Not moving fast enough, maintain current yaw
                desired_yaw = self.current_yaw
        elif (self.target_distance < 3.0 and 
              self.navigation_mode == self.NAVIGATION_MODE_VISUAL):
            # Very close in visual mode - use the exact orientation from target
            # The target_yaw already contains the vision-computed orientation
            desired_yaw = self.target_yaw
        else:
            # Medium distance - face toward the target
            desired_yaw = math.atan2(pos_error[1], pos_error[0])
            
        # Calculate yaw error
        yaw_error = desired_yaw - self.current_yaw
        
        # Normalize to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
            
        return yaw_error
        
    def set_offboard_mode(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        
        # Publish zero velocity until mode is set
        zero_vel = TwistStamped()
        zero_vel.header.frame_id = "odom"
        
        for _ in range(100):  # Try for 5 seconds at 20Hz
            if self.mavros_current_state.mode == "OFFBOARD":
                rospy.loginfo("OFFBOARD mode set successfully")
                return True
                
            zero_vel.header.stamp = rospy.Time.now()
            self.vel_pub.publish(zero_vel)
            
            if rospy.Time.now().to_sec() % 1.0 < 0.05:  # Every second
                self.set_mode_client.call(offb_set_mode)
                
            self.rate.sleep()
            
        rospy.logwarn("Failed to set OFFBOARD mode")
        return False
        
    def run(self):
        # Wait for FCU connection
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.mavros_current_state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected")
        
        # Set initial offboard mode
        self.set_offboard_mode()
        
        while not rospy.is_shutdown():
            if self.mavros_current_state.mode != "OFFBOARD" and self.current_state != self.STATE_HOLDING_POSITION:
                rospy.logerr("Not in OFFBOARD mode. Attempting to set...")
                self.set_offboard_mode()
                
            if self.current_state in [self.STATE_NAVIGATING, self.STATE_APPROACHING]:
                self.control_loop()
            elif self.current_state == self.STATE_HOLDING_POSITION:
                self.hold_position()
                
            self.rate.sleep()
            
    def control_loop(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        if dt <= 0 or dt > 1.0:  # Sanity check
            return
            
        # Calculate position error
        pos_error = self.target_pos - self.current_pos
        distance = np.linalg.norm(pos_error)
        self.target_distance = distance
        
        # Check if goal reached
        tolerance = (self.visual_goal_tolerance if self.navigation_mode == self.NAVIGATION_MODE_VISUAL 
                    else self.goal_tolerance)
        
        if distance <= tolerance:
            rospy.loginfo(f"Goal reached! Distance: {distance:.3f}m")
            self.goal_reached_pub.publish(Bool(True))
            self.switch_state(self.STATE_HOLDING_POSITION)
            self.set_mode(self.hover_mode)
            return
            
        # Update adaptive gains and get max speed
        max_speed = self.update_adaptive_gains()
        
        # Calculate velocity profile
        if self.enable_velocity_profiling:
            current_speed = np.linalg.norm(self.current_vel)
            desired_speed = self.calculate_velocity_profile(distance, current_speed)
            max_speed = min(max_speed, desired_speed)
            
        # Compute hybrid yaw control
        yaw_error = self.compute_hybrid_yaw_rate(pos_error)
        
        # PID control
        error = np.array([pos_error[0], pos_error[1], pos_error[2], yaw_error])
        
        # Integral with anti-windup
        self.integral += error * dt
        integral_limit = 2.0  # Adjust as needed
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        
        # Derivative with filtering
        if np.any(self.previous_error):
            derivative = (error - self.previous_error) / dt
            # Simple low-pass filter on derivative
            alpha = 0.7
            derivative = alpha * derivative + (1 - alpha) * (self.previous_error / dt)
        else:
            derivative = np.zeros(4)
            
        # PID output
        output = (self.current_kp * error + 
                 self.integral * self.ki_cruise +  # Use base ki for now
                 self.current_kd * derivative)
        
        self.previous_error = error
        
        # Extract linear and angular components
        linear_output = output[:3]
        
        # Apply speed limit
        speed = np.linalg.norm(linear_output)
        if speed > max_speed:
            linear_output = (linear_output / speed) * max_speed
            
        # Limit yaw rate
        yaw_rate = np.clip(output[3], -self.max_yaw_rate, self.max_yaw_rate)
        
        # Publish velocity command
        vel_msg = TwistStamped()
        vel_msg.header.stamp = current_time
        vel_msg.header.frame_id = "odom"
        vel_msg.twist.linear.x = linear_output[0]
        vel_msg.twist.linear.y = linear_output[1]
        vel_msg.twist.linear.z = linear_output[2]
        vel_msg.twist.angular.z = yaw_rate
        
        self.vel_pub.publish(vel_msg)
        
        # Log status
        rospy.loginfo_throttle(1.0, 
            f"{self.current_state}: Dist={distance:.2f}m, Speed={speed:.2f}m/s, "
            f"Yaw={self.yaw_mode}, Mode={self.navigation_mode}")
            
    def hold_position(self):
        """Publish zero velocity to hold position"""
        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = "odom"
        self.vel_pub.publish(vel_msg)
        
    def set_mode(self, mode):
        try:
            req = SetModeRequest()
            req.custom_mode = mode
            resp = self.set_mode_client.call(req)
            return resp.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
            
    def shutdown_cb(self):
        rospy.loginfo("Shutting down PID controller")
        self.set_mode(self.hover_mode)
        
        # Send zero velocity
        zero_vel = TwistStamped()
        zero_vel.header.frame_id = "odom"
        for _ in range(20):
            zero_vel.header.stamp = rospy.Time.now()
            self.vel_pub.publish(zero_vel)
            rospy.sleep(0.05)
            
if __name__ == '__main__':
    try:
        controller = EnhancedGoalToVelocityPID()
        controller.run()
    except rospy.ROSInterruptException:
        pass