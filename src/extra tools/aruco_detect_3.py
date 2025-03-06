#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Set the hardcoded ArUCo tag type
hardcoded_aruco_type = "DICT_ARUCO_ORIGINAL"

# Define ARUCO_DICT to map ArUco types to OpenCV's ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Function to draw detected markers on the image (replace aruco_display)
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (marker_corner, marker_id) in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            # Convert the (x, y) coordinates to integers
            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            # Draw the bounding box of the ArUCo marker
            cv2.line(image, top_left, top_right, (0, 255, 0), 2)
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

            # Draw the marker ID on the image
            cv2.putText(image, str(marker_id), (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Callback function for image topics
def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(e)
        return

    h, w, _ = cv_image.shape
    width = 600
    height = int(width * (h / w))
    cv_image = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(hardcoded_aruco_type, None) is None:
        print(f"ArUCo tag type '{hardcoded_aruco_type}' is not supported")
        return

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    print("Detecting '{}' tags....".format(hardcoded_aruco_type))
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[hardcoded_aruco_type])
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=aruco_params)

    detected_markers_1 = aruco_display(corners, ids, rejected, cv_image)
    cv2.imshow("Image", detected_markers_1)
    cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("aruco_detector", anonymous=True)
    bridge = CvBridge()
    image_topic_1= "/iris/downward_camera/downward_camera/rgb/image_raw"
    image_topic_2= "/iris/camera/rgb/image_raw"

    rospy.Subscriber(image_topic_1, Image, image_callback)
    rospy.Subscriber(image_topic_2, Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
