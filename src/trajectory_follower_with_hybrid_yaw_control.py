#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import threading
import tf.transformations
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode, SetModeRequest
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectory
import math
from std_msgs.msg import Bool

class PVATrajectoryFollower:
    def __init__(self):
        rospy.init_node('pva_trajectory_follower')

        self.lock = threading.Lock()

        # PID gains for yaw control
        self.kp_yaw = rospy.get_param('~kp_yaw', 0.8)
        self.ki_yaw = rospy.get_param('~ki_yaw', 0.1)
        self.kd_yaw = rospy.get_param('~kd_yaw', 0.2)
        self.switch_distance = rospy.get_param('~yaw_switch_distance', 2.0)
        self.align_with_velocity_distance = rospy.get_param('~align_with_velocity_distance', 10.0)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 1.0) # rad/s

        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1)
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.1)
        self.is_gps_navigation_mode = rospy.get_param('~is_gps_navigation_mode', False)

        self.current_pose = None
        self.current_vel = np.zeros(3)
        self.current_yaw = 0.0
        self.current_orientation_q = None # New: Store current orientation quaternion
        
        self.latest_pva = None
        self.goal_point = None

        # PID state for yaw control
        self.yaw_error_integral = 0.0
        self.last_yaw_error = 0.0
        self.last_time = None

        # Visualization Publisher
        self.path_pub = rospy.Publisher('/pva_hybrid_yaw_path', Path, queue_size=1)

        self.mavros_current_state = State()
        rospy.Subscriber('/mavros/state', State, self.mavros_state_callback)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # Subscribers
        rospy.Subscriber('/pva_setpoint', JointTrajectoryPoint, self.pva_setpoint_cb)
        rospy.Subscriber('/command/trajectory', MultiDOFJointTrajectory, self.trajectory_vis_cb)
        rospy.Subscriber('/goal_point', PoseStamped, self.goal_point_cb)
        rospy.Subscriber('/uav/odometry', Odometry, self.odom_cb)

        self.target_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/drone_events/reached_desired_pose', Bool, queue_size=1)
        self.gps_goal_reached_pub = rospy.Publisher('/drone_events/reached_goal_point', Bool, queue_size=1)
        
        
        self.rate = rospy.Rate(50)
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.loginfo("Waiting for first PVA setpoint on /pva_setpoint...")
        while not rospy.is_shutdown() and self.latest_pva is None:
            rospy.sleep(0.1)
        rospy.loginfo("First PVA setpoint received.")

        rospy.loginfo("Waiting for goal point on /goal_point...")
        while not rospy.is_shutdown() and self.goal_point is None:
            rospy.sleep(0.1)
        rospy.loginfo("Goal point received.")

        self.set_offboard_mode()
        self.control_loop()

    def mavros_state_callback(self, msg):
        self.mavros_current_state = msg

    def pva_setpoint_cb(self, msg):
        # Callback for single PVA point, used for control
        with self.lock:
            self.latest_pva = msg

    def trajectory_vis_cb(self, msg):
        # This callback visualizes the upcoming trajectory segment with predicted yaw
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"

        for point in msg.points:
            future_pose = np.array([point.transforms[0].translation.x,
                                    point.transforms[0].translation.y,
                                    point.transforms[0].translation.z])
            future_vel = np.array([point.velocities[0].linear.x,
                                   point.velocities[0].linear.y,
                                   point.velocities[0].linear.z])

            desired_yaw = self.calculate_future_yaw(future_pose, future_vel)

            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp # Use same stamp for all poses in path
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = future_pose[0]
            pose.pose.position.y = future_pose[1]
            pose.pose.position.z = future_pose[2]
            
            q = tf.transformations.quaternion_from_euler(0, 0, desired_yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def goal_point_cb(self, msg):
        with self.lock:
            self.goal_point = msg

    def odom_cb(self, msg):
        with self.lock:
            self.current_pose = np.array([msg.pose.pose.position.x,
                                          msg.pose.pose.position.y,
                                          msg.pose.pose.position.z])
            self.current_vel = np.array([msg.twist.twist.linear.x,
                                         msg.twist.twist.linear.y,
                                         msg.twist.twist.linear.z])
            orientation_q = msg.pose.pose.orientation
            self.current_orientation_q = orientation_q # Store the quaternion
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
            self.current_yaw = yaw

    def calculate_future_yaw(self, future_pose, future_vel):
        # This is a helper to predict yaw for a future point in a trajectory
        if self.goal_point is None:
            return 0.0

        goal_pos = np.array([self.goal_point.pose.position.x,
                             self.goal_point.pose.position.y,
                             self.goal_point.pose.position.z])
        
        distance_to_goal = np.linalg.norm(future_pose - goal_pos)

        if distance_to_goal > self.align_with_velocity_distance:
            if np.linalg.norm(future_vel) > 0.1:
                desired_yaw = math.atan2(future_vel[1], future_vel[0])
            else:
                direction_vector = goal_pos - future_pose
                desired_yaw = math.atan2(direction_vector[1], direction_vector[0])
        elif distance_to_goal > self.switch_distance:
            direction_vector = goal_pos - future_pose
            desired_yaw = math.atan2(direction_vector[1], direction_vector[0])
        else:
            orientation_q = self.goal_point.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, desired_yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        
        return desired_yaw

    def set_offboard_mode(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        last_req = rospy.Time.now()
        
        rospy.loginfo("Attempting to set OFFBOARD mode...")
        for i in range(100):
            if rospy.is_shutdown():
                return
            if self.latest_pva:
                self.publish_pva_to_mavros(0.0) # Publish with zero yaw rate
            self.rate.sleep()

        while not rospy.is_shutdown() and self.mavros_current_state.mode != "OFFBOARD":
            if (rospy.Time.now() - last_req) > rospy.Duration(2.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled.")
                else:
                    rospy.logwarn_throttle(5, "Failed to set OFFBOARD mode.")
                last_req = rospy.Time.now()
            
            if self.latest_pva:
                self.publish_pva_to_mavros(0.0) # Publish with zero yaw rate
            self.rate.sleep()
        rospy.loginfo("Drone is in OFFBOARD mode.")

    def calculate_yaw_rate(self):
        if self.goal_point is None or self.current_pose is None:
            return 0.0, 0.0 # Return a tuple of two floats

        goal_pos = np.array([self.goal_point.pose.position.x,
                             self.goal_point.pose.position.y,
                             self.goal_point.pose.position.z])
        
        distance_to_goal = np.linalg.norm(self.current_pose - goal_pos)

        if distance_to_goal > self.align_with_velocity_distance:
            # Align with the velocity vector
            if np.linalg.norm(self.current_vel) > 0.1:
                desired_yaw = math.atan2(self.current_vel[1], self.current_vel[0])
            else:
                # Default to looking at the goal if not moving
                direction_vector = goal_pos - self.current_pose
                desired_yaw = math.atan2(direction_vector[1], direction_vector[0])
        elif distance_to_goal > self.switch_distance:
            # Look towards the goal point
            direction_vector = goal_pos - self.current_pose
            desired_yaw = math.atan2(direction_vector[1], direction_vector[0])
        else:
            # Use the yaw from the goal point's orientation
            orientation_q = self.goal_point.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, desired_yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        # PID controller for yaw rate
        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            return 0.0, 0.0 # Return a tuple of two floats
        
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            return 0.0, 0.0 # Return a tuple of two floats

        error = desired_yaw - self.current_yaw
        # Normalize angle error to [-pi, pi]
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        self.yaw_error_integral += error * dt
        # Optional: Add integral windup protection
        self.yaw_error_integral = max(min(self.yaw_error_integral, 1.0), -1.0)

        derivative = (error - self.last_yaw_error) / dt
        
        yaw_rate = self.kp_yaw * error + self.ki_yaw * self.yaw_error_integral + self.kd_yaw * derivative

        # Apply clipping to yaw_rate
        yaw_rate = max(min(yaw_rate, self.max_yaw_rate), -self.max_yaw_rate)

        self.last_yaw_error = error
        self.last_time = current_time

        return yaw_rate, error

    def publish_pva_to_mavros(self, yaw_rate):
        target_msg = PositionTarget()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.header.frame_id = "odom"

        # Set the coordinate frame to LOCAL_ENU (value 1)
        target_msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        # We want to control P, V, A, and Yaw Rate, so we ignore Yaw Angle.
        target_msg.type_mask = PositionTarget.IGNORE_YAW
        
        target_msg.position.x = self.latest_pva.positions[0]
        target_msg.position.y = self.latest_pva.positions[1]
        target_msg.position.z = self.latest_pva.positions[2]
        
        target_msg.velocity.x = self.latest_pva.velocities[0]
        target_msg.velocity.y = self.latest_pva.velocities[1]
        target_msg.velocity.z = self.latest_pva.velocities[2]
        
        target_msg.acceleration_or_force.x = self.latest_pva.accelerations[0]
        target_msg.acceleration_or_force.y = self.latest_pva.accelerations[1]
        target_msg.acceleration_or_force.z = self.latest_pva.accelerations[2]
        
        target_msg.yaw_rate = yaw_rate

        self.target_pub.publish(target_msg)

    def control_loop(self):
        self.last_time = rospy.Time.now()
        while not rospy.is_shutdown():
            with self.lock:
                if self.current_pose is None or self.latest_pva is None or self.goal_point is None:
                    self.rate.sleep()
                    continue
                
                yaw_rate, yaw_error = self.calculate_yaw_rate()
                self.publish_pva_to_mavros(yaw_rate)

                # Check if goal is reached and publish status
                goal_pos = np.array([self.goal_point.pose.position.x,
                                     self.goal_point.pose.position.y,
                                     self.goal_point.pose.position.z])
                distance_to_goal = np.linalg.norm(self.current_pose - goal_pos)

                if distance_to_goal < self.position_tolerance and abs(yaw_error) < self.yaw_tolerance:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(True)
                        rospy.loginfo_throttle(1, "GPS Goal Reached: Position and Yaw within tolerance.")
                    else:
                        self.goal_reached_pub.publish(True)
                        rospy.loginfo_throttle(1, "Visual Servoing Goal Reached: Position and Yaw within tolerance.")
                else:
                    if self.is_gps_navigation_mode:
                        self.gps_goal_reached_pub.publish(False)
                    else:
                        self.goal_reached_pub.publish(False)

            self.rate.sleep()

    def shutdown_hook(self):
        rospy.loginfo(f"{rospy.get_name()} is shutting down. Setting mode to AUTO.LOITER.")
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
    try:
        controller = PVATrajectoryFollower()
    except rospy.ROSInterruptException:
        pass
