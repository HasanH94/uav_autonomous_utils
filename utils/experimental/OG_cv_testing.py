#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from tf.transformations import euler_from_matrix, euler_from_quaternion

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.velocity_topic = "/mavros/setpoint_velocity/cmd_vel_unstamped" #self.velocity_topic = "/mavros/setpoint_velocity/cmd_vel"
        self.yaw = 0.0
        self.body_x_vel = 0.0
        self.body_y_vel = 0.0
        self.body_z_vel = 0.0
        self.body_yaw_vel = 0.0
        self.bridge = CvBridge()

        # MAVROS related attributes
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.local_pos_pub = rospy.Publisher(self.velocity_topic, Twist, queue_size=10) #self.local_pos_pub = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=10)

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # Load calibration data
        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)

        # Load camera parameters
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters()

        # Initialize publisher and subscriber for images
        rospy.Subscriber(self.image_topic, Image, self.callback_function)

        # Offboard and arming setup
        self.rate = rospy.Rate(20)
        self.pose = Twist()  #TwistStamped()
        self.last_req = rospy.Time.now()

        self.integral_error_x = 0
        self.integral_error_y = 0
        self.integral_error_z = 0
        self.integral_error_yaw = 0

        # Initialize error variables for derivative control
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_z = 0.0
        self.prev_error_yaw = 0.0
        self.dt = 0.05  # Time step for derivative control

        # Ensure FCU connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        # Send a few setpoints before starting offboard mode
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = 'OFFBOARD'

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

    def state_cb(self, msg):
        self.current_state = msg
    
    def pose_callback(self, pose_msg):
        # Get the orientation quaternion
        orientation_q = pose_msg.pose.orientation

        # Convert quaternion to Euler angles
        _, _, self.yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])


    def callback_function(self, image_msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect markers
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        if marker_IDs is not None and len(marker_IDs) > 0:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, 8, self.cam_mat, self.dist_coef)
            
            for i in range(len(marker_IDs)):
                # Calculate the position of the marker in the camera frame
                marker_position = tVec[i][0]  # [x, y, z] of the marker
                
                # Define the desired position (0, 0, 1.5) in the camera frame
                desired_position = np.array([0, 0, 1.5])
                
                # Calculate position errors
                error_x_meters = (desired_position[0] - marker_position[0])/10
                error_y_meters = (desired_position[1] - marker_position[1])/10
                distance_z = -(desired_position[2] - marker_position[2])/10  # z distance from the marker to the drone

                def normalize_angle(angle):
                    if angle<np.pi:
                        return angle
                    return -(2*np.pi - angle)


                # Yaw control: Get yaw from rVec and compute yaw error
                rotation_matrix, _ = cv.Rodrigues(rVec[i])
                yaw_marker = euler_from_matrix(rotation_matrix)[2]  # Extract yaw (rotation around z-axis)
                desired_yaw = np.pi  # Desired yaw (aligned with marker)
                yaw_error = normalize_angle(desired_yaw - yaw_marker) # Calculate yaw error

                # Proportional control gains
                k_x = 0.018
                k_y = 0.2
                k_z = 0.2
                k_yaw = 0.2 # Gain for yaw control (tune based on your system)

                #k_x = 0.00
                #k_y = 0.0
                #k_z = 0.0
                #k_yaw = 0.0

                kd_x = 0.0
                kd_y = 0.0
                kd_z = 0.0
                kd_yaw = 0.0

                integral_gain_x = 0.0  # Adjust as necessary
                integral_gain_y = 0.0 # Adjust as necessary
                integral_gain_z = 0.0  # Adjust as necessary
                integral_gain_yaw = 0.0 # Adjust as necessary

                self.integral_error_x += error_x_meters * self.dt
                self.integral_error_y += error_y_meters * self.dt
                self.integral_error_z += distance_z * self.dt
                self.integral_error_yaw += yaw_error * self.dt 

                # Calculate derivative terms
                derivative_x = (error_x_meters - self.prev_error_x) / self.dt
                derivative_y = (error_y_meters - self.prev_error_y) / self.dt
                derivative_z = (distance_z - self.prev_error_z) / self.dt
                derivative_yaw = (yaw_error - self.prev_error_yaw) / self.dt  # Now yaw_error is defined

                # Calculate the velocity commands based on errors and derivative terms
                # self.pose.twist.linear.x = k_x * distance_z + kd_x * derivative_z + (integral_gain_x * self.integral_error_z)
                # self.pose.twist.linear.y = k_y * error_x_meters + kd_y * derivative_x + (integral_gain_y * self.integral_error_x)
                # self.pose.twist.linear.z = k_z * error_y_meters + kd_z * derivative_y + + (integral_gain_z * self.integral_error_y)
                # self.pose.twist.angular.z = k_yaw * yaw_error + kd_yaw * derivative_yaw + (integral_gain_yaw * self.integral_error_yaw)

                # Saturate the command values
                # max_speed = 10
                # self.pose.twist.linear.x = max(min(self.pose.twist.linear.x, max_speed), -max_speed)
                # self.pose.twist.linear.y = max(min(self.pose.twist.linear.y, max_speed), -max_speed)
                # self.pose.twist.linear.z = max(min(self.pose.twist.linear.z, max_speed), -max_speed)

                # Update previous errors for the next iteration
                self.prev_error_x = error_x_meters
                self.prev_error_y = error_y_meters
                self.prev_error_z = distance_z
                self.prev_error_yaw = yaw_error

                # Publish the command
                self.local_pos_pub.publish(self.pose)

                # Visual feedback
                distance = np.sqrt(marker_position[0]**2 + marker_position[1]**2 + marker_position[2]**2)
                cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rVec[i], tVec[i], 4, 4)

                # Display error information in meters
                cv.putText(
                    frame,
                    f"x: {round(error_x_meters, 3)}m y: {round(error_y_meters, 3)}m z: {round(distance_z, 3)}m",
                    (10, 90), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )
                cv.putText(
                    frame,
                    f"yaw_error: {round(yaw_error, 3)} rad",
                    (10, 110), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )

        # Add red dot at the center of the frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv.imshow("frame", frame)
        cv.waitKey(1)

        # Ensure offboard mode and arming
        if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - self.last_req) > rospy.Duration(5.0):
            if self.set_mode_client.call(self.offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            self.last_req = rospy.Time.now()
        elif not self.current_state.armed and (rospy.Time.now() - self.last_req) > rospy.Duration(5.0):
            if self.arming_client.call(self.arm_cmd).success:
                rospy.loginfo("Vehicle armed")
            self.last_req = rospy.Time.now()
        
        self.pose.linear.y = 1


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        aruco_detector = ArucoDetectorNode()
        aruco_detector.run()
    except rospy.ROSInterruptException:
        pass
