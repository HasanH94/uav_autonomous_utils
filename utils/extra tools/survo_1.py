#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node')
        self.image_topic = "/iris/camera/rgb/image_raw"
        self.velocity_topic = "/mavros/setpoint_velocity/cmd_vel"
        self.bridge = CvBridge()

        # MAVROS related attributes
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.local_pos_pub = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=10)

        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

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

        # Initialize publisher and subscriber for images
        rospy.Subscriber(self.image_topic, Image, self.callback_function)

        # Offboard and arming setup
        self.rate = rospy.Rate(20)
        self.pose = TwistStamped()
        self.last_req = rospy.Time.now()

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
                desired_position = np.array([0, 0, 1.5])  # Adjust based on your requirements
                
                # Calculate errors
                error_x_meters = desired_position[0] - marker_position[0]
                error_y_meters = desired_position[1] - marker_position[1]
                distance_z = -(desired_position[2] - marker_position[2])  # z distance from the marker to the drone

                # Proportional control gains
                k_x = 0.02
                k_y = 0.01
                k_z = 0.01
                
                # Calculate the velocity commands based on errors
                self.pose.twist.linear.x = k_x * distance_z
                self.pose.twist.linear.y = k_y * error_x_meters
                self.pose.twist.linear.z = k_z * error_y_meters
                
                # Saturate the command values
                max_speed = 4.0
                self.pose.twist.linear.x = max(min(self.pose.twist.linear.x, max_speed), -max_speed)
                self.pose.twist.linear.y = max(min(self.pose.twist.linear.y, max_speed), -max_speed)
                self.pose.twist.linear.z = max(min(self.pose.twist.linear.z, max_speed), -max_speed)

                # Stop condition based on error thresholds
                error_threshold = 0.01
                if (abs(error_x_meters) < error_threshold and
                    abs(error_y_meters) < error_threshold and
                    abs(distance_z - 1.5) < error_threshold):
                    self.pose.twist.linear.x = 0
                    self.pose.twist.linear.y = 0
                    self.pose.twist.linear.z = 0

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

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        aruco_detector = ArucoDetectorNode()
        aruco_detector.run()
    except rospy.ROSInterruptException:
        pass
