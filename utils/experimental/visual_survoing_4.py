#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np
from geometry_msgs.msg import TwistStamped  # Import TwistStamped message type for velocity commands

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.velocity_topic = "/mavros/local_position/velocity_local"  # Topic to publish velocity commands
        self.bridge = CvBridge()

        # Load calibration data
        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)
        print(self.calib_data.files)

        # Load camera parameters
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()

        # Get focal lengths from camera matrix
        self.focal_length_x = self.cam_mat[0, 0]  # fx
        self.focal_length_y = self.cam_mat[1, 1]  # fy

        # Initialize publisher and subscriber
        self.velocity_pub = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self.callback_function)

    def callback_function(self, image_msg):
        #print('callback function called_1!')
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect markers
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        if marker_IDs is not None and len(marker_IDs) > 0:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, 8, self.cam_mat, self.dist_coef)
            
            for i in range(len(marker_IDs)):
                #print("callback function called_2!")
                # Calculate the position of the marker in the camera frame
                marker_position = tVec[i][0]  # [x, y, z] of the marker
                
                # Define the desired position (0, 0, 1.5) in the camera frame
                desired_position = np.array([0, 0, 1.5])  # Adjust based on your requirements
                
                # Calculate errors
                error_x_meters = desired_position[0] - marker_position[0]
                error_y_meters = desired_position[1] - marker_position[1]
                distance_z = -(desired_position[2] - marker_position[2])  # z distance from the marker to the drone
                #print(error_x_meters,error_y_meters,distance_z,end=",")
                cmd = TwistStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.header.frame_id = "base_link"

                # Proportional control gains
                k_x = 5.0  # Increase gain for x direction
                k_y = 1.0  # Increase gain for y direction
                k_z = 1.0  # Increase gain for z direction
                
                # Calculate the velocity commands based on errors
                cmd.twist.linear.x = k_x * distance_z  # Changed here
                cmd.twist.linear.y = k_y * error_x_meters  # Changed here
                cmd.twist.linear.z = k_z * error_y_meters  # Changed here
                
                # Saturate the command values
                max_speed = 10.0  # Define a maximum speed
                #cmd.twist.linear.x = max(min(cmd.twist.linear.x, max_speed), -max_speed)  # Changed here
                #cmd.twist.linear.y = max(min(cmd.twist.linear.y, max_speed), -max_speed)  # Changed here
                #cmd.twist.linear.z = max(min(cmd.twist.linear.z, max_speed), -max_speed)  # Changed here

                # Stop condition based on error thresholds
                error_threshold = 0.01  # Set a threshold for stopping
                if (abs(error_x_meters) < error_threshold and
                    abs(error_y_meters) < error_threshold and
                    abs(distance_z - 1.5) < error_threshold):
                    cmd.twist.linear.x = 0  # Changed here
                    cmd.twist.linear.y = 0  # Changed here
                    cmd.twist.linear.z = 0  # Changed here

                # Publish the command
                self.velocity_pub.publish(cmd)

                # Visual feedback
                distance = np.sqrt(marker_position[0]**2 + marker_position[1]**2 + marker_position[2]**2)
                cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rVec[i], tVec[i], 4, 4)

                # Extract rotation matrix from rotation vector
                rotation_matrix = cv.Rodrigues(rVec[i])[0]

                # Extract Euler angles (roll, pitch, yaw) from rotation matrix
                euler_angles = cv.RQDecomp3x3(rotation_matrix)[0]

                # Display rotation information
                cv.putText(
                    frame,
                    f"Rotation: (Roll: {round(euler_angles[0], 2)}, Pitch: {round(euler_angles[1], 2)}, Yaw: {round(euler_angles[2], 2)})",
                    (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )

                # Display error information in meters
                cv.putText(
                    frame,
                    f"ID: {marker_IDs[i][0]} Dist: {round(distance_z, 2)}m",
                    (10, 60), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )
                cv.putText(
                    frame,
                    f"x: {round(error_x_meters, 3)}m y: {round(error_y_meters, 3)}m z: {round(distance_z, 3)}m",
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
