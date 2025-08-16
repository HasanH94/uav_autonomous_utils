#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.bridge = CvBridge()

        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)

        # Subscribe to drone state
        self.current_state = State()
        rospy.Subscriber("/mavros/state", State, self.state_callback)

        # Service client for setting flight mode
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # Load calibration data
        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)

        # Load camera parameters
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()

        # Subscribe to the image topic
        rospy.Subscriber(self.image_topic, Image, self.callback_function)

        # Target distance (in meters)
        self.target_distance = 0
        # Define gains for the control algorithm
        self.kp_x = 1.0  # Proportional gain for x control
        self.kp_y = 1.0  # Proportional gain for y control
        self.kp_z = 1.0  # Proportional gain for z control

        # Maximum velocity (m/s)
        self.max_velocity = 2.0

        # Initialize previous errors for derivative control
        self.prev_x_error = 0.0
        self.prev_y_error = 0.0
        self.prev_z_error = 0.0

        # Store the current frame
        self.frame = None

    def state_callback(self, state_msg):
        self.current_state = state_msg

    def set_offboard_mode(self):
        if self.current_state.mode != "OFFBOARD":
            if self.set_mode_client.call(custom_mode="OFFBOARD").mode_sent:
                rospy.loginfo("OFFBOARD mode enabled")
            else:
                rospy.logwarn("Failed to set OFFBOARD mode")

    def visual_servoing_control(self, tVec, corners):
        # Ensure we're in OFFBOARD mode
        self.set_offboard_mode()

        # Get image dimensions
        image_height, image_width = self.frame.shape[:2]
        image_center_x = image_width / 2
        image_center_y = image_height / 2

        # Calculate marker center in image coordinates
        marker_center_x = (corners[0][0][0] + corners[0][2][0]) / 2
        marker_center_y = (corners[0][0][1] + corners[0][2][1]) / 2

        # Calculate error in pixels
        x_error = (marker_center_x - image_center_x) / image_width  # Normalize to [-0.5, 0.5]
        y_error = (marker_center_y - image_center_y) / image_height  # Normalize to [-0.5, 0.5]
        z_error = self.target_distance - tVec[2]  # Z-axis error (forward/backward)

        # Create a Twist message for velocity commands
        cmd = Twist()

        # Compute control commands using proportional control
        cmd.linear.x = -self.kp_z * z_error  # Forward/Backward
        cmd.linear.y = self.kp_x * x_error   # Left/Right
        cmd.linear.z = -self.kp_y * y_error   # Up/Down

        # Limit the velocity commands
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_velocity, self.max_velocity)
        cmd.linear.y = np.clip(cmd.linear.y, -self.max_velocity, self.max_velocity)
        cmd.linear.z = np.clip(cmd.linear.z, -self.max_velocity, self.max_velocity)

        # Log the error values for debugging
        rospy.loginfo(f"Errors - x:{x_error:.2f}, y:{y_error:.2f}, z:{z_error:.2f}")
        rospy.loginfo(f"Published velocity command - Linear: x:{cmd.linear.x:.2f}, y:{cmd.linear.y:.2f}, z:{cmd.linear.z:.2f}")

        # Publish the velocity command
        self.cmd_pub.publish(cmd)

    def callback_function(self, image_msg):
        self.frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, 0.08, self.cam_mat, self.dist_coef)  # Marker size is 8cm
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv.polylines(self.frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)

                # Control logic
                self.visual_servoing_control(tVec[i][0], corners)

                # Other information (ID, distance, x, y)
                distance = np.linalg.norm(tVec[i][0])
                cv.putText(self.frame, f"ID: {ids[0]} Dist: {distance:.2f}m", (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv.putText(self.frame, f"x: {tVec[i][0][0]:.2f} y: {tVec[i][0][1]:.2f} z: {tVec[i][0][2]:.2f}", (10, 60), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # Add red dot at the center of the frame
        center_x = self.frame.shape[1] // 2
        center_y = self.frame.shape[0] // 2
        cv.circle(self.frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv.imshow("frame", self.frame)
        cv.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        aruco_detector = ArucoDetectorNode()
        aruco_detector.run()
    except rospy.ROSInterruptException:
        pass
