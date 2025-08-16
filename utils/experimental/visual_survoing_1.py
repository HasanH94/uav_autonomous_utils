#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np
from geometry_msgs.msg import Twist

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('visual_servoing_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.bridge = CvBridge()

        # Load calibration data
        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)
        print(self.calib_data.files)

        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()

        rospy.Subscriber(self.image_topic, Image, self.callback_function)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Desired distance from the ArUco marker
        self.target_distance = 1.5

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
                # Calculate distance and orientation
                distance = np.linalg.norm(tVec[i][0])
                x_error = tVec[i][0][0]
                y_error = tVec[i][0][1]
                z_error = tVec[i][0][2] - self.target_distance  # Z-axis error for distance control

                # Proportional control (adjust to fit your needs)
                control = Twist()
                control.linear.x = 0.1 * z_error  # Move forward/backward based on distance error
                control.linear.y = 500 * x_error  # Move left/right based on x-axis error
                control.linear.z = 0.5 * y_error  # Move up/down based on y-axis error

                self.vel_pub.publish(control)  # Publish control commands

                # Display ArUco marker and distance on the frame
                cv.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )
                cv.putText(
                    frame,
                    f"Distance: {round(distance, 2)}m",
                    (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (200, 100, 0), 2, cv.LINE_AA
                )

        # Show the frame with marker info
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
