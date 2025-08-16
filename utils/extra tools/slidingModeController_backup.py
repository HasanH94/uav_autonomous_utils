#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import threading
import tf.transformations
from std_msgs.msg import Bool
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State

class SlidingModeVelocityController:
    def __init__(self):
        rospy.init_node('sliding_mode_vel_controller')
        self.lambda_ = rospy.get_param('~lambda', 1.0) # Parameter for sliding surface
        self.gamma = rospy.get_param('~gamma', 0.1) # Parameter for saturation function
        self.mass = rospy.get_param('~mass', 1.0)
        self.max_speed = rospy.get_param('~max_speed', 2.5)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 2.0) # Max yaw rate in rad/s
        self.g = 9.81
        self.yaw_switch_distance = rospy.get_param('~yaw_switch_distance', 1.0) # Distance to goal to switch yaw control

        # Backstepping SMC parameters (to be tuned based on paper)
        self.c1 = rospy.get_param('~c1', 1.0) # For position error
        self.c2 = rospy.get_param('~c2', 1.0) # For attitude error
        self.k_s = rospy.get_param('~k_s', 0.5) # Gain for robust term
        self.eta = rospy.get_param('~eta', 0.1) # Gain for sign function

        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1) # Position tolerance for goal reached
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.1) # Yaw tolerance for goal reached
        self.is_gps_navigation_mode = rospy.get_param('~is_gps_navigation_mode', False) # True if used for GPS navigation

        self.goal = None
        self.goal_orientation = None
        self.current_pose = None
        self.current_vel = np.zeros(3)
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_yaw_rate = 0.0

        self.mavros_current_state = State()
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb)
        self.pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/drone_events/reached_desired_pose', Bool, queue_size=1)
        self.gps_goal_reached_pub = rospy.Publisher('/drone_events/reached_goal_point', Bool, queue_size=1)

        self.lock = threading.Lock()
        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.shutdown_hook)
        self.control_loop()

    def mavros_state_callback(self, msg):
        self.mavros_current_state = msg

    def set_offboard_mode(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        last_req = rospy.Time.now()

        rospy.loginfo("Attempting to set OFFBOARD mode...")
        # Send a few setpoints before attempting to switch mode
        # This is crucial for PX4 to accept OFFBOARD mode
        for i in range(100):
            if rospy.is_shutdown():
                return
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = "odom"
            self.pub.publish(twist)
            self.rate.sleep()

        while not rospy.is_shutdown() and self.mavros_current_state.mode != "OFFBOARD":
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled.")
                else:
                    rospy.logwarn_throttle(5, "Failed to set OFFBOARD mode.")
                last_req = rospy.Time.now()

            # Publish setpoints even while waiting for OFFBOARD mode to be set
            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = "odom"
            self.pub.publish(twist)
            self.rate.sleep()
        rospy.loginfo("Drone is in OFFBOARD mode.")

    def sat(self, s):
        return np.clip(s / self.gamma, -1.0, 1.0)

    def control_loop(self):
        while not rospy.is_shutdown():
            with self.lock:
                if self.goal is None or self.current_pose is None:
                    self.rate.sleep()
                    continue

                # Position Errors (e = x_d - x)
                e_x = self.goal[0] - self.current_pose[0]
                e_y = self.goal[1] - self.current_pose[1]
                e_z = self.goal[2] - self.current_pose[2]

                # Velocity Errors (edot = x_dot_d - x_dot)
                # Assuming desired velocity is 0 for a static goal
                edot_x = -self.current_vel[0]
                edot_y = -self.current_vel[1]
                edot_z = -self.current_vel[2]

                # Sliding Surfaces for Position (s = edot + lambda * e)
                s_x = edot_x + self.lambda_ * e_x
                s_y = edot_y + self.lambda_ * e_y
                s_z = edot_z + self.lambda_ * e_z

                # --- Backstepping Step 1: Position Control to get Desired Thrust (U1) and Desired Attitude ---
                # Desired Total Thrust (U1) from z-axis control (simplified from paper's Eq. 54 for Sz)
                # U1 = (m / (cos(phi)cos(theta))) * (g + z_ddot_d + lambda * edot_z + k_s * sat(s_z) + eta * s_z)
                # Assuming z_ddot_d = 0 for static goal
                # Note: The paper's full U1 includes more terms, this is a simplified version for velocity control
                U1_desired = (self.mass / (np.cos(self.current_roll) * np.cos(self.current_pitch))) * \
                             (self.g + self.lambda_ * edot_z + self.k_s * self.sat(s_z) + self.eta * s_z)
                
                # Desired Roll (phi_d) and Pitch (theta_d) from x and y position control
                # These are virtual control inputs
                # Based on simplified dynamics: x_ddot = (cos(phi)sin(theta)cos(psi) + sin(phi)sin(psi)) * U1 / m
                # y_ddot = (cos(phi)sin(theta)sin(psi) - sin(phi)cos(psi)) * U1 / m
                # We want x_ddot_desired = -lambda * edot_x - k_s * sat(s_x) - eta * s_x
                # And y_ddot_desired = -lambda * edot_y - k_s * sat(s_y) - eta * s_y

                # Calculate desired accelerations from x and y sliding surfaces
                x_ddot_desired = -self.lambda_ * edot_x - self.k_s * self.sat(s_x) - self.eta * s_x
                y_ddot_desired = -self.lambda_ * edot_y - self.k_s * self.sat(s_y) - self.eta * s_y

                # Solve for desired roll and pitch
                # This is a simplified inversion of the quadrotor's translational dynamics
                # Assuming small angles for simplicity in initial implementation
                # From paper's Eq. 1, simplified:
                # x_ddot = (theta*cos(psi) + phi*sin(psi)) * U1 / m
                # y_ddot = (theta*sin(psi) - phi*cos(psi)) * U1 / m
                
                # This is a system of linear equations for phi and theta:
                # [sin(psi)  cos(psi)] [phi] = [x_ddot_desired * m / U1]
                # [-cos(psi) sin(psi)] [theta] = [y_ddot_desired * m / U1]
                
                # Solving for phi_d and theta_d:
                if U1_desired > 0.01: # Avoid division by zero
                    term_x = x_ddot_desired * self.mass / U1_desired
                    term_y = y_ddot_desired * self.mass / U1_desired
                    
                    # This is the inverse transformation for small angles
                    phi_d = term_x * np.sin(self.current_yaw) - term_y * np.cos(self.current_yaw)
                    theta_d = term_x * np.cos(self.current_yaw) + term_y * np.sin(self.current_yaw)
                else:
                    phi_d = 0.0
                    theta_d = 0.0

                # --- Backstepping Step 2: Attitude Control to get Desired Angular Velocities (Torques) ---
                # Attitude Errors (e_att = att_d - att)
                e_roll = phi_d - self.current_roll
                e_pitch = theta_d - self.current_pitch
                
                # Yaw error (hybrid control)
                horizontal_dist = np.linalg.norm(self.goal[:2] - self.current_pose[:2])
                desired_yaw = self.current_yaw # Default to current yaw if no goal or close enough

                if horizontal_dist > self.yaw_switch_distance:
                    # Far from goal: point towards the goal position
                    delta_x = self.goal[0] - self.current_pose[0]
                    delta_y = self.goal[1] - self.current_pose[1]
                    if abs(delta_x) > 0.01 or abs(delta_y) > 0.01: # Avoid division by zero or erratic behavior near goal
                        desired_yaw = np.arctan2(delta_y, delta_x)
                elif self.goal_orientation is not None:
                    # Close to goal: align with the published goal orientation
                    goal_orientation_list = [self.goal_orientation.x, self.goal_orientation.y, self.goal_orientation.z, self.goal_orientation.w]
                    (roll_g, pitch_g, desired_yaw) = tf.transformations.euler_from_quaternion(goal_orientation_list)
                
                e_yaw = desired_yaw - self.current_yaw
                e_yaw = np.arctan2(np.sin(e_yaw), np.cos(e_yaw)) # Normalize yaw error
                rospy.loginfo(f"Yaw Error: {e_yaw:.4f}")

                # Angular Velocity Errors (edot_att = att_dot_d - att_dot)
                # Assuming desired angular velocities are 0 for static attitude goals
                edot_roll = 0.0 - 0.0 # Assuming roll rate is not directly available from odom, or desired is 0
                edot_pitch = 0.0 - 0.0 # Assuming pitch rate is not directly available from odom, or desired is 0
                edot_yaw = 0.0 - self.current_yaw_rate

                # Sliding Surfaces for Attitude
                s_roll = edot_roll + self.c2 * e_roll
                s_pitch = edot_pitch + self.c2 * e_pitch
                s_yaw = edot_yaw + self.c2 * e_yaw

                # Desired Torques (U2, U3, U4)
                # These are simplified from paper's Eq. 54 for attitude control
                # Assuming Ixx, Iyy, Izz are 1 for simplicity in initial implementation
                # U2 (roll torque) = Ixx * (phi_ddot_d + c2 * edot_roll + k_s * sat(s_roll) + eta * s_roll)
                # U3 (pitch torque) = Iyy * (theta_ddot_d + c2 * edot_pitch + k_s * sat(s_pitch) + eta * s_pitch)
                # U4 (yaw torque) = Izz * (psi_ddot_d + c2 * edot_yaw + k_s * sat(s_yaw) + eta * s_yaw)
                # Assuming desired angular accelerations are 0

                # For now, let's map these directly to angular velocities for TwistStamped
                # This is a simplification, a proper mapping would involve drone's inertia and dynamics
                
                # Linear velocity commands (from U1_desired, assuming it's primarily for z-axis velocity)
                # This is a very rough mapping. A proper backstepping controller would output forces/torques
                # which then need to be converted to motor commands, and then to linear/angular velocities.
                # For now, let's use U1_desired as a proxy for desired vertical velocity, and phi_d/theta_d for horizontal
                
                # This part needs careful thought: how to map forces/torques to TwistStamped velocities.
                # The paper's output is U1, U2, U3, U4 (thrust and torques).
                # We need to convert these to linear.x, linear.y, linear.z, angular.z
                
                # Let's assume U1_desired directly translates to desired linear.z velocity
                # And desired roll/pitch translate to desired linear.x/y velocities (simplified)
                # And desired yaw torque translates to desired angular.z velocity
                
                # Linear velocities
                twist = TwistStamped()
                twist.header.stamp = rospy.Time.now()
                twist.header.frame_id = "odom"

                # For linear velocities, we can use the desired accelerations from position control
                # and integrate them, or directly use the error terms.
                # Given the previous controller was velocity-based, let's try to derive velocities.
                # The output of the backstepping controller is forces/torques.
                # We need to map these to velocities. This is a common challenge.
                
                # Let's try to map the sliding surfaces directly to velocities, scaled by lambda
                # This is similar to the previous controller's output (edot + lambda*e)
                
                # Linear velocities based on position sliding surfaces
                twist.twist.linear.x = s_x
                twist.twist.linear.y = s_y
                twist.twist.linear.z = s_z

                # Log position error
                total_position_error = np.linalg.norm(np.array([e_x, e_y, e_z]))
                rospy.loginfo(f"Total Position Error: {total_position_error:.4f}")

                # Apply max_speed limit
                linear_speed = np.linalg.norm([twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z])
                if linear_speed > self.max_speed:
                    twist.twist.linear.x = (twist.twist.linear.x / linear_speed) * self.max_speed
                    twist.twist.linear.y = (twist.twist.linear.y / linear_speed) * self.max_speed
                    twist.twist.linear.z = (twist.twist.linear.z / linear_speed) * self.max_speed

                # Angular Z velocity based on yaw sliding surface
                twist.twist.angular.z = s_yaw
                # Apply max_yaw_rate limit
                if abs(twist.twist.angular.z) > self.max_yaw_rate:
                    twist.twist.angular.z = np.sign(twist.twist.angular.z) * self.max_yaw_rate

                self.pub.publish(twist)

                # Check if goal is reached and publish status
                if total_position_error < self.position_tolerance and abs(e_yaw) < self.yaw_tolerance:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(True)
                        rospy.loginfo("GPS Goal Reached: Position and Yaw within tolerance.")
                    else:
                        self.goal_reached_pub.publish(True)
                        rospy.loginfo("Visual Servoing Goal Reached: Position and Yaw within tolerance.")
                else:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(False)
                    else:
                        self.goal_reached_pub.publish(False)

            self.rate.sleep()

    def goal_cb(self, msg):
        with self.lock:
            self.goal = np.array([msg.pose.position.x,
                                  msg.pose.position.y,
                                  msg.pose.position.z])
            self.goal_orientation = msg.pose.orientation

    def odom_cb(self, msg):
        with self.lock:
            self.current_pose = np.array([msg.pose.pose.position.x,
                                          msg.pose.pose.position.y,
                                          msg.pose.pose.position.z])
            self.current_vel = np.array([msg.twist.twist.linear.x,
                                         msg.twist.twist.linear.y,
                                         msg.twist.twist.linear.z])
            # Extract yaw from quaternion
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
            self.current_roll = roll
            self.current_pitch = pitch
            self.current_yaw = yaw
            self.current_yaw_rate = msg.twist.twist.angular.z

    def shutdown_hook(self):
        rospy.loginfo("SlidingModeVelocityController is shutting down. Setting mode to AUTO.LOITER.")
        set_mode_srv = SetModeRequest()
        set_mode_srv.custom_mode = 'AUTO.LOITER'
        try:
            if self.set_mode_client.call(set_mode_srv).mode_sent:
                rospy.loginfo("Successfully set to AUTO.LOITER mode.")
            else:
                rospy.logwarn("Failed to set AUTO.LOITER mode on shutdown.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed on shutdown: {e}")

if __name__ == '__main__':
    Slider = SlidingModeVelocityController()
