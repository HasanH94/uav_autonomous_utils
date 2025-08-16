#!/usr/bin/env python3


import numpy as np
#from utils import ARUCO_DICT, aruco_display
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Set the hardcoded ArUCo tag type
hardcoded_aruco_type = "DICT_ARUCO_ORIGINAL"

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

    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(hardcoded_aruco_type, None) is None:
        print(f"ArUCo tag type '{hardcoded_aruco_type}' is not supported")
        return

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    print("Detecting '{}' tags....".format(hardcoded_aruco_type))
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[hardcoded_aruco_type])
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=aruco_params)

    detected_markers_1 = aruco_display(corners, ids, rejected, cv_image)
    detected_markers_2 = aruco_display(corners, ids, rejected, cv_image)
    cv2.imshow("Image", detected_markers_1)
    cv2.imshow("Image", detected_markers_2)
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
