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

        # Update the publisher to use the unstamped topic
        self.cmd_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)

        # Subscribe to drone state
        self.current_state = State()
        rospy.Subscriber("/mavros/state", State, self.state_callback)

        # Service client for setting flight mode
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # Load calibration data
        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)
        print(self.calib_data.files)

        # Load camera parameters
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()

        # Subscribe to the image topic
        rospy.Subscriber(self.image_topic, Image, self.callback_function)

        # Set target distance (in meters)
        self.target_distance = 1.5
        # Define gains for the control algorithm
        self.kp_x = 0.5  # Proportional gain for x control
        self.kp_y = 0.5  # Proportional gain for y control
        self.kp_z = 0.5  # Proportional gain for z control

        # Define maximum velocity (m/s)
        self.max_velocity = 1.0

    def state_callback(self, state_msg):
        self.current_state = state_msg

    def set_offboard_mode(self):
        if self.current_state.mode != "OFFBOARD":
            if self.set_mode_client.call(custom_mode="OFFBOARD").mode_sent:
                rospy.loginfo("OFFBOARD mode enabled")
            else:
                rospy.logwarn("Failed to set OFFBOARD mode")

    def visual_servoing_control(self, tVec, distance):
        # Ensure we're in OFFBOARD mode
        self.set_offboard_mode()

        # Calculate the error in the position
        x_error = tVec[0]  # X-axis error
        y_error = tVec[1]  # Y-axis error
        z_error = tVec[2] - self.target_distance  # Z-axis error

        # Create a Twist message for velocity commands
        cmd = Twist()

        # Compute control commands using proportional control
        cmd.linear.x = self.kp_x * x_error
        cmd.linear.y = self.kp_y * y_error
        cmd.linear.z = self.kp_z * z_error

        # Limit the velocity commands
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_velocity, self.max_velocity)
        cmd.linear.y = np.clip(cmd.linear.y, -self.max_velocity, self.max_velocity)
        cmd.linear.z = np.clip(cmd.linear.z, -self.max_velocity, self.max_velocity)

        # Publish the velocity command
        self.cmd_pub.publish(cmd)
        
        # Add logging
        rospy.loginfo(f"Published velocity command - Linear: x:{cmd.linear.x:.2f}, y:{cmd.linear.y:.2f}, z:{cmd.linear.z:.2f}")

    def callback_function(self, image_msg):
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, self.marker_dict, parameters=self.param_markers
        )

        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, 8, self.cam_mat, self.dist_coef
            )
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )

                distance = np.sqrt(
                    tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2 + tVec[i][0][2] ** 2
                )

                # Control logic
                self.visual_servoing_control(tVec[i][0], distance)

                # Other information (ID, distance, x, y)
                cv.putText(
                    frame,
                    f"ID: {ids[0]} Dist: {round(distance, 2)}",
                    (10, 60), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )
                cv.putText(
                    frame,
                    f"x: {round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)}",
                    (10, 90), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )

        # Add red dot at the center of the frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv.imshow("frame", frame)
        cv.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        aruco_detector = ArucoDetectorNode()
        aruco_detector.run()
    except rospy.ROSInterruptException:
        pass