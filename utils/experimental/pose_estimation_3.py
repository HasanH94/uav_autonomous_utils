#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.bridge = CvBridge()

        calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        self.calib_data = np.load(calib_data_path)
        print(self.calib_data.files)

        #Load All PARAMETERS
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()

        rospy.Subscriber(self.image_topic, Image, self.callback_function)

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
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                bottom_right = corners[1].ravel()
                bottom_left = corners[2].ravel()

                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                point = cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rVec[i], tVec[i], 4, 4)

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

                # Other information (ID, distance, x, y)
                cv.putText(
                    frame,
                    f"ID: {ids[0]} Dist: {round(tVec[i][0][2], 2)}",
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
