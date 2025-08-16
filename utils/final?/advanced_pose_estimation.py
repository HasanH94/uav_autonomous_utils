#!/usr/bin/env python3
import rospy
import cv2 as cv
from cv2 import aruco
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_matrix

class PoseEstimationNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')

        # Topics
        self.image_topic = "/iris_obs_avoid/camera/rgb/image_raw"
        self.detection_status_topic = "/marker_detection_status"
        self.pose_topic = "/aruco_single/pose"  # This topic will publish the robust pose

        self.bridge = CvBridge()
        self.current_state = State()

        # MAVROS state subscriber (if needed)
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        # Get marker length from a ROS parameter (physical marker size in meters)
        self.marker_length = rospy.get_param("~marker_length", 0.15)

        # Camera parameters (from your Gazebo model)
        image_width = 848    # extracted from SDF
        image_height = 480   # extracted from SDF
        horizontal_fov = 1.5009831567  # in radians, from SDF

        # Calculate focal length assuming a pinhole camera model
        focal_length = (image_width / 2) / np.tan(horizontal_fov / 2)

        # Construct the camera matrix and assume zero distortion (from SDF)
        self.cam_mat = np.array([[focal_length, 0, image_width / 2],
                                 [0, focal_length, image_height / 2],
                                 [0, 0, 1]])
        self.dist_coef = np.zeros((5,))  # [k1, k2, p1, p2, k3]

        # ArUco marker settings
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        # Use the DetectorParameters constructor instead of DetectorParameters_create()
        self.param_markers = aruco.DetectorParameters()

        # Subscribers and publishers
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=10)
        self.detection_status_pub = rospy.Publisher(self.detection_status_topic, Bool, queue_size=10)

    def state_cb(self, msg):
        self.current_state = msg

    def image_callback(self, image_msg):
        # Convert the ROS image message to an OpenCV BGR image
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect markers in the grayscale image
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        # Prepare the detection status message
        detection_status = Bool()

        if marker_IDs is not None and len(marker_IDs) > 0:
            detection_status.data = True

            # Use the first detected marker for pose estimation
            i = 0
            # marker_corners[i] is an array with shape (1, 4, 2); reshape to (4,2)
            corners = marker_corners[i].reshape(-1, 2)

            # Define the 3D object points for the marker corners.
            # Assuming the marker is a square centered at (0,0,0) with z=0,
            # and the corner order corresponds to [top-left, top-right, bottom-right, bottom-left]:
            half_size = self.marker_length / 2.0
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)

            # Use RANSAC-based PnP to estimate the pose
            ret, rvec, tvec, inliers = cv.solvePnPRansac(object_points, corners, self.cam_mat, self.dist_coef)
            if not ret:
                rospy.logwarn("solvePnPRansac failed")
                detection_status.data = False
            else:
                # Optionally, scale the translation by 10 (as determined by your tests)
                tvec_scaled = tvec * 10

                # Convert the rotation vector to a rotation matrix
                R, _ = cv.Rodrigues(rvec)

                # Build a homogeneous transformation matrix (4x4)
                T = np.eye(4)
                T[:3, :3] = R

                # Convert the rotation matrix to a quaternion
                quat = quaternion_from_matrix(T)

                # Create a PoseStamped message with the robust pose
                pose_msg = PoseStamped()
                pose_msg.header = image_msg.header  # Use the same header (timestamp & frame_id)
                pose_msg.pose.position.x = tvec_scaled[0][0]
                pose_msg.pose.position.y = tvec_scaled[1][0]
                pose_msg.pose.position.z = tvec_scaled[2][0]
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                self.pose_pub.publish(pose_msg)

                # Draw the marker coordinate axes on the frame for visualization
                cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rvec, tvec, 0.1)

        else:
            detection_status.data = False
            rospy.loginfo("No ArUco marker detected.")

        # Publish detection status (True if a marker is detected, False otherwise)
        self.detection_status_pub.publish(detection_status)

        # Optionally, display the frame with overlays for debugging
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
