#!/usr/bin/env python3
import rospy
import cv2 as cv
from cv2 import aruco
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from std_msgs.msg import Bool  # Import Bool message type
from tf.transformations import euler_from_matrix

class PoseEstimationNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        #self.image_topic = "/iris_obs_avoid/camera/rgb/image_raw"   #"/iris/camera/rgb/image_raw"
        self.image_topic = "/iris_obs_avoid/camera/rgb/image_raw"
        self.visual_error_topic = "/visual_errors"
        self.detection_status_topic = "/marker_detection_status"  # New topic for detection status
        self.bridge = CvBridge()
        self.current_state = State()

        # MAVROS state subscriber
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        # Calibration data loading
        #calib_data_path = "/home/hasan/research_ws/src/othmanPack/src/MultiMatrix.npz"
        #self.calib_data = np.load(calib_data_path)
        #self.cam_mat = self.calib_data["camMatrix"]
        #self.dist_coef = self.calib_data["distCoef"]
        # Print camera matrix and distortion coefficients
        #print("Camera Matrix:\n" + np.array2string(self.cam_mat) + "\nDistortion Coefficients:\n" + np.array2string(self.dist_coef))
        
        # Camera parameters from the Gazebo model
        image_width = 848  # extracted from SDF
        image_height = 480  # extracted from SDF
        horizontal_fov = 1.5009831567  # in radians, extracted from SDF

        # Calculate the focal length
        focal_length = (image_width / 2) / np.tan(horizontal_fov / 2)

        # Camera matrix
        self.cam_mat = np.array([[focal_length, 0, image_width / 2],
                            [0, focal_length, image_height / 2],
                            [0, 0, 1]])

        # Distortion coefficients (since they are all zeros in the SDF)
        self.dist_coef = np.zeros((5,))  # Assuming standard distortion coefficients: [k1, k2, p1, p2, k3]

        # Aruco marker settings
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters()

        # Image subscriber
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        
        # Visual errors publisher
        self.visual_error_pub = rospy.Publisher(self.visual_error_topic, Twist, queue_size=10)

        # Detection status publisher
        self.detection_status_pub = rospy.Publisher(self.detection_status_topic, Bool, queue_size=10)

    def state_cb(self, msg):
        self.current_state = msg

    def image_callback(self, image_msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect markers
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        # Initialize variables to avoid referencing before assignment
        error_x_meters, error_y_meters, distance_z, yaw_error = 0, 0, 0, 0

        # Initialize detection status
        detection_status = Bool()
        
        if marker_IDs is not None and len(marker_IDs) > 0:
            detection_status.data = True  # Marker detected
            
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, 8, self.cam_mat, self.dist_coef)

            for i in range(len(marker_IDs)):
                # Marker position
                factor = 5
                marker_position = tVec[i][0]  # [x, y, z] of the marker
                desired_position = np.array([0, 0, 1.5*factor])  # Target position

                # Calculate position errors
                error_x_meters = (desired_position[0] - marker_position[0]) / factor
                error_y_meters = (desired_position[1] - marker_position[1]) / factor
                distance_z = -(desired_position[2] - marker_position[2]) / factor  # Z distance

                # Yaw control: Get yaw from rVec and compute yaw error
                rotation_matrix, _ = cv.Rodrigues(rVec[i])
                yaw_marker = euler_from_matrix(rotation_matrix)[2]  # Extract yaw
                desired_yaw = np.pi  # Desired yaw (aligned with marker)

                def normalize_angle(angle):
                    if angle < np.pi:
                        return angle
                    return -(2 * np.pi - angle)

                yaw_error = normalize_angle(desired_yaw - yaw_marker)

                # Publish errors
                error_msg = Twist()
                error_msg.linear.x = error_x_meters
                error_msg.linear.y = error_y_meters
                error_msg.linear.z = distance_z
                error_msg.angular.z = yaw_error
                self.visual_error_pub.publish(error_msg)

                # Draw the marker axis
                cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rVec[i], tVec[i], 4, 4)

        else:
            detection_status.data = False  # No marker detected
            rospy.loginfo("No ArUco marker detected.")

        # Publish detection status
        self.detection_status_pub.publish(detection_status)

        # Display the error information on the frame even if no marker is detected (can show zeros)
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

        # Show the frame
        cv.imshow("Pose Estimation", frame)
        cv.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PoseEstimationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
