#!/usr/bin/env python3
import rospy
import cv2
from cv2 import aruco
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_matrix
import message_filters
import math

class ArucoGoalPublisher:
    def __init__(self):
        rospy.init_node('aruco_goal_publisher')

        # --- Parameters ---
        self.standoff_distance = rospy.get_param('~standoff_distance', 1.2) # meters
        self.target_marker_id = rospy.get_param('~target_marker_id', 1) # Only consider this marker ID
        self.target_frame = rospy.get_param('~target_frame', 'odom')
        self.enable_visualization = rospy.get_param('~enable_visualization', False)
        self.max_drone_to_goal_distance = rospy.get_param('~max_drone_to_goal_distance', 100.0) # meters
        self.min_marker_pixel_area = rospy.get_param('~min_marker_pixel_area', 1000) # pixels^2
        self.max_marker_pixel_area = rospy.get_param('~max_marker_pixel_area', 100000) # pixels^2
        
        # --- ROS Hooks ---
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- State Variables ---
        self.cam_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None
        self.current_drone_pose = None # To store the drone's current pose
        self.mission_goal = None # To store the state machine's goal
        self.focal_x = 0.0
        self.focal_y = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.debug_image = None
        self.debug_depth_image = None # Initialize debug depth image

        # --- Aruco Setup ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters()
        # Make the detector much stricter to reduce false positives
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.polygonalApproxAccuracyRate = 0.02 # Default is 0.05, very strict
        self.aruco_params.adaptiveThreshWinSizeMax = 23 # Default is 23
        self.aruco_params.adaptiveThreshWinSizeMin = 3 # Default is 3
        self.aruco_params.adaptiveThreshConstant = 10.0 # Default is 7.0, a bit stricter
        self.aruco_params.minOtsuStdDev = 5.0 # Default is 5.0
        self.aruco_params.errorCorrectionRate = 0.4 # Default is 0.6, less error tolerant

        # --- Publishers ---
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.marker_found_pub = rospy.Publisher('/drone_flags/marker_found', Bool, queue_size=1)
        self.marker_pose_pub = rospy.Publisher('~/marker_pose_debug', PoseStamped, queue_size=1)

        # --- Initialization ---
        if not self.wait_for_camera_info():
            rospy.logerr("Could not get camera info. Shutting down.")
            rospy.signal_shutdown("Camera info not available.")
            return

        # --- Subscribers ---
        self.image_sub = message_filters.Subscriber('/iris/camera/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/iris/camera/depth/image_raw', Image)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.drone_pose_callback)
        rospy.Subscriber('/move_base_mission/goal', PoseStamped, self.mission_goal_callback)

        rospy.loginfo(f"ArucoGoalPublisher initialized. Waiting for mission goal. Visualization: {self.enable_visualization}")

    def mission_goal_callback(self, msg):
        rospy.loginfo_once("Received first mission goal.")
        self.mission_goal = msg

    def wait_for_camera_info(self):
        rospy.loginfo("Waiting for camera info...")
        try:
            cam_info_msg = rospy.wait_for_message('/iris/camera/rgb/camera_info', CameraInfo, timeout=5.0)
            rospy.loginfo("Received camera calibration info.")
            self.cam_matrix = np.array(cam_info_msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(cam_info_msg.D)
            self.camera_frame = cam_info_msg.header.frame_id
            self.focal_x = self.cam_matrix[0, 0]
            self.focal_y = self.cam_matrix[1, 1]
            self.center_x = self.cam_matrix[0, 2]
            self.center_y = self.cam_matrix[1, 2]
            return True
        except rospy.ROSException:
            rospy.logwarn("Timeout while waiting for CameraInfo message.")
            return False

    def drone_pose_callback(self, msg):
        self.current_drone_pose = msg

    def image_callback(self, rgb_msg, depth_msg):
        # Ensure drone pose is available
        if self.current_drone_pose is None:
            rospy.logwarn_throttle(1.0, "Waiting for drone's current pose...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Log image dimensions for debugging
            rospy.loginfo_throttle(5.0, f"Image Dimensions - RGB: {cv_image.shape[1]}x{cv_image.shape[0]}, Depth: {depth_image.shape[1]}x{depth_image.shape[0]}")

            if cv_image is None or depth_image is None:
                rospy.logwarn_throttle(1.0, "Failed to convert image messages to OpenCV format.")
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            # DEBUG: Log raw detection results
            if ids is not None:
                rospy.loginfo_throttle(1.0, f"Raw detected IDs: {ids.flatten()}")
            else:
                rospy.loginfo_throttle(1.0, "Raw detected IDs: None")

            tvec_display = np.zeros(3)
            marker_detected_flag = False
            detected_ids_str = "None"
            best_marker_id = -1

            if ids is not None:
                # Filter out None IDs and create a list of (id, original_index) tuples
                valid_markers = []
                for i, marker_id_arr in enumerate(ids):
                    marker_id = marker_id_arr[0] # ids is a list of arrays, each containing one ID
                    if marker_id == self.target_marker_id: # Only consider the target marker ID
                        # Calculate pixel area of the detected marker
                        marker_corners = corners[i][0]
                        area = cv2.contourArea(marker_corners)

                        # --- Aspect Ratio Check ---
                        # Get the bounding rectangle to calculate aspect ratio
                        rect = cv2.minAreaRect(marker_corners)
                        (x, y), (w, h), angle = rect
                        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                        if self.min_marker_pixel_area <= area <= self.max_marker_pixel_area and aspect_ratio < 1.2:
                            valid_markers.append((marker_id, i, area))
                        else:
                            rospy.logwarn_throttle(1.0, f"Target marker ID {marker_id} detected but failed validation. Area: {area:.2f}, Aspect Ratio: {aspect_ratio:.2f}. Ignoring.")
                    else:
                        rospy.logwarn_throttle(1.0, f"Non-target marker ID {marker_id} detected. Ignoring.")

                if len(valid_markers) > 0:
                    marker_detected_flag = True
                    detected_ids_str = ", ".join(map(str, [f"{m[0]} (Area: {m[2]:.0f})" for m in valid_markers]))

                    # --- Pre-computation for all valid markers ---
                    all_marker_poses = {}
                    for marker_id, original_index, _ in valid_markers:
                        rvec, tvec = self.get_marker_pose(corners[original_index], cv_image.shape, depth_image)
                        if tvec is None: continue
                        marker_pose_odom = self.transform_to_odom(rvec, tvec, rgb_msg.header.stamp)
                        if marker_pose_odom is None: continue
                        all_marker_poses[original_index] = {
                            'id': marker_id,
                            'rvec': rvec,
                            'tvec': tvec,
                            'odom_pose': marker_pose_odom,
                            'area': area
                        }
                        rospy.loginfo_throttle(1.0, f"  [DEBUG] Detected Valid Marker ID: {marker_id}, Cam Pose: (x:{tvec[0]:.2f}, y:{tvec[1]:.2f}, z:{tvec[2]:.2f}), Odom Pose: (x:{marker_pose_odom.pose.position.x:.2f}, y:{marker_pose_odom.pose.position.y:.2f}, z:{marker_pose_odom.pose.position.z:.2f})")

                    # Decide which marker is the target
                    if len(all_marker_poses) == 1:
                        # If only one valid marker, it's our target
                        target_index = next(iter(all_marker_poses))
                        best_marker_id = all_marker_poses[target_index]['id']
                    elif self.mission_goal is not None:
                        # If multiple markers are seen and we have a mission goal, find the closest one.
                        rospy.loginfo_throttle(1.0, "Multiple valid markers detected. Finding closest to mission goal.")
                        closest_dist = float('inf')
                        for original_index, pose_data in all_marker_poses.items():
                            marker_pose_odom = pose_data['odom_pose']
                            dist = np.linalg.norm([
                                marker_pose_odom.pose.position.x - self.mission_goal.pose.position.x,
                                marker_pose_odom.pose.position.y - self.mission_goal.pose.position.y,
                                marker_pose_odom.pose.position.z - self.mission_goal.pose.position.z
                            ])
                            if dist < closest_dist:
                                closest_dist = dist
                                best_marker_id = pose_data['id']
                                target_index = original_index
                    else:
                        # If multiple markers and no mission goal, pick the one with the largest pixel area (highest confidence)
                        rospy.logwarn_throttle(1.0, "Multiple valid markers detected but no mission goal. Picking largest marker.")
                        largest_area = -1
                        for marker_id, original_index, area in valid_markers:
                            if area > largest_area:
                                largest_area = area
                                best_marker_id = marker_id
                                target_index = original_index

                else: # No valid markers found after filtering
                    marker_detected_flag = False
                    detected_ids_str = "None (no valid IDs)"
                    best_marker_id = -1 # Ensure no marker is selected

            if best_marker_id != -1 and target_index in all_marker_poses:
                rospy.loginfo_throttle(1.0, f"Target marker {best_marker_id} selected!")

                # Retrieve the pre-computed pose data
                pose_data = all_marker_poses[target_index]
                rvec = pose_data['rvec']
                tvec = pose_data['tvec']

                tvec_display = tvec # For visualization

                rospy.loginfo_throttle(1.0, f"ARUCO tvec (camera frame): {tvec}")
                rospy.loginfo_throttle(1.0, f"ARUCO rvec (rotation vector): {rvec}")

                # Convert rvec to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # --- Sanity Check for Pose Estimation (Camera Frame) ---
                current_distance_to_marker = np.linalg.norm(tvec)
                rospy.loginfo_throttle(1.0, f"Current distance to marker: {current_distance_to_marker:.2f}m")
                rospy.loginfo_throttle(1.0, f"Standoff distance: {self.standoff_distance:.2f}m")

                if tvec[2] < 0 or current_distance_to_marker > 100.0: # Marker behind camera or too far away
                    rospy.logwarn_throttle(1.0, f"Invalid marker pose (z={tvec[2]:.2f}m, dist={current_distance_to_marker:.2f}m). Not publishing goal.")
                    marker_detected_flag = False # Override flag for visualization
                elif current_distance_to_marker <= self.standoff_distance and self.standoff_distance > 0: # Only warn if standoff is positive
                    rospy.logwarn_throttle(1.0, f"Drone is too close (dist: {current_distance_to_marker:.2f}m). Goal not published.")
                else:
                    # The goal is defined in the marker's frame, `standoff_distance` along the marker's positive Z-axis.
                    goal_in_marker_frame = np.array([0.0, 0.0, self.standoff_distance])
                    rospy.loginfo_throttle(1.0, f"Goal in marker frame: {goal_in_marker_frame}")

                    # Transform goal from marker frame to camera frame
                    # P_camera = R * P_marker + tvec
                    goal_position_in_camera_frame = R.dot(goal_in_marker_frame) + tvec
                    rospy.loginfo_throttle(1.0, f"Goal position in camera frame: {goal_position_in_camera_frame}")
                    if self.standoff_distance == 0.0:
                        rospy.loginfo_throttle(1.0, "Standoff distance is 0.0. Goal position in camera frame should be equal to ARUCO tvec.")

                    goal_pose_camera = PoseStamped()
                    goal_pose_camera.header.stamp = rgb_msg.header.stamp
                    goal_pose_camera.header.frame_id = self.camera_frame
                    goal_pose_camera.pose.position.x = goal_position_in_camera_frame[0]
                    goal_pose_camera.pose.position.y = goal_position_in_camera_frame[1]
                    goal_pose_camera.pose.position.z = goal_position_in_camera_frame[2]
                    
                    # To make the drone face the marker, we want the drone's X-axis to point along the marker's -Z axis.
                    R_face_marker = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
                    R_goal = R.dot(R_face_marker)

                    # Create a 4x4 homogeneous transformation matrix from the new rotation matrix
                    T_goal_to_camera = np.identity(4)
                    T_goal_to_camera[:3, :3] = R_goal

                    # Convert rotation matrix to quaternion
                    q = quaternion_from_matrix(T_goal_to_camera)

                    goal_pose_camera.pose.orientation.x = q[0]
                    goal_pose_camera.pose.orientation.y = q[1]
                    goal_pose_camera.pose.orientation.z = q[2]
                    goal_pose_camera.pose.orientation.w = q[3]

                    self.marker_pose_pub.publish(goal_pose_camera)

                    try:
                        transform = self.tf_buffer.lookup_transform(self.target_frame, self.camera_frame, rgb_msg.header.stamp, rospy.Duration(1.0))
                        rospy.loginfo_throttle(1.0, f"Transform from {self.camera_frame} to {self.target_frame}:")
                        rospy.loginfo_throttle(1.0, f"  Translation: x={transform.transform.translation.x:.2f}, y={transform.transform.translation.y:.2f}, z={transform.transform.translation.z:.2f}")
                        rospy.loginfo_throttle(1.0, f"  Rotation: x={transform.transform.rotation.x:.2f}, y={transform.transform.rotation.y:.2f}, z={transform.transform.rotation.z:.2f}, w={transform.transform.rotation.w:.2f}")
                        goal_pose_odom = tf2_geometry_msgs.do_transform_pose(goal_pose_camera, transform)
                        rospy.loginfo_throttle(1.0, f"Goal pose in ODOM frame (position): x={goal_pose_odom.pose.position.x:.2f}, y={goal_pose_odom.pose.position.y:.2f}, z={goal_pose_odom.pose.position.z:.2f}")
                        rospy.loginfo_throttle(1.0, f"Goal pose in ODOM frame (orientation): x={goal_pose_odom.pose.orientation.x:.2f}, y={goal_pose_odom.pose.orientation.y:.2f}, z={goal_pose_odom.pose.orientation.z:.2f}, w={goal_pose_odom.pose.orientation.w:.2f}")
                        
                        # --- Sanity Check: Distance from Drone to Goal ---
                        if self.current_drone_pose:
                            drone_to_goal_distance = np.linalg.norm([
                                goal_pose_odom.pose.position.x - self.current_drone_pose.pose.position.x,
                                goal_pose_odom.pose.position.y - self.current_drone_pose.pose.position.y,
                                goal_pose_odom.pose.position.z - self.current_drone_pose.pose.position.z
                            ])
                            if drone_to_goal_distance > self.max_drone_to_goal_distance:
                                rospy.logwarn_throttle(1.0, f"Drone-to-goal distance too large ({drone_to_goal_distance:.2f}m). Not publishing goal.")
                            else:
                                self.goal_pub.publish(goal_pose_odom)
                                rospy.loginfo_throttle(1.0, f"Publishing goal with standoff distance of {self.standoff_distance}m.")
                        else:
                            rospy.logwarn_throttle(1.0, "Drone pose not available for drone-to-goal distance check.")

                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logwarn(f"Could not transform from {self.camera_frame} to {self.target_frame}: {e}")
            
            if self.enable_visualization:
                # Draw detected markers and their axes
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                if best_marker_id != -1: # Only draw axes if the target marker was found
                    # We need rvec and tvec of the best marker for drawing, so we re-calculate it if necessary
                    if 'rvec' not in locals() or 'tvec' not in locals():
                        target_index = np.where(ids == best_marker_id)[0][0]
                        rvec, tvec = self.get_marker_pose(corners[target_index], depth_image)
                    if tvec is not None:
                        cv2.drawFrameAxes(cv_image, self.cam_matrix, self.dist_coeffs, rvec, tvec, 0.15 * 1.5, 2)

                    # Draw dots on all four corners of the detected marker for verification
                    marker_corners = corners[target_index][0]
                    # Top-left corner (index 0)
                    cv2.circle(cv_image, (int(marker_corners[0][0]), int(marker_corners[0][1])), 5, (255, 0, 0), -1) # Blue dot
                    # Top-right corner (index 1)
                    cv2.circle(cv_image, (int(marker_corners[1][0]), int(marker_corners[1][1])), 5, (0, 255, 0), -1) # Green dot
                    # Bottom-right corner (index 2)
                    cv2.circle(cv_image, (int(marker_corners[2][0]), int(marker_corners[2][1])), 5, (0, 0, 255), -1) # Red dot
                    # Bottom-left corner (index 3)
                    cv2.circle(cv_image, (int(marker_corners[3][0]), int(marker_corners[3][1])), 5, (0, 255, 255), -1) # Yellow dot

                # Display detection status
                status_text = f"Marker Found: {marker_detected_flag} (Target: {best_marker_id})"
                cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(cv_image, f"Detected IDs: {detected_ids_str}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Display pose info
                cv2.putText(cv_image, f"x:{tvec_display[0]:.2f} y:{tvec_display[1]:.2f} z:{tvec_display[2]:.2f}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw red dot at center of the marker
                # Get the center of the marker in RGB pixel coordinates for visualization
                rgb_cx_vis = int(np.mean(marker_corners[:, 0]))
                rgb_cy_vis = int(np.mean(marker_corners[:, 1]))

                # Draw a yellow rectangle on the 10x10 window used for depth calculation
                win_size_vis = 10 # This should match the win_size in get_marker_pose
                x_start_rect = max(0, rgb_cx_vis - win_size_vis // 2)
                y_start_rect = max(0, rgb_cy_vis - win_size_vis // 2)
                x_end_rect = min(cv_image.shape[1], rgb_cx_vis + win_size_vis // 2)
                y_end_rect = min(cv_image.shape[0], rgb_cy_vis + win_size_vis // 2)
                cv2.rectangle(cv_image, (x_start_rect, y_start_rect), (x_end_rect, y_end_rect), (0, 255, 255), 2) # Yellow rectangle
                
                self.debug_image = cv_image

                # --- Visualize Depth Window on Depth Image ---
                if best_marker_id != -1:
                    # Get the center of the marker in RGB pixel coordinates
                    marker_corners = corners[target_index][0]
                    rgb_cx = int(np.mean(marker_corners[:, 0]))
                    rgb_cy = int(np.mean(marker_corners[:, 1]))

                    # Get image dimensions for scaling
                    rgb_h, rgb_w, _ = cv_image.shape
                    depth_h, depth_w = depth_image.shape

                    # Calculate scaling factors
                    scale_x = depth_w / float(rgb_w)
                    scale_y = depth_h / float(rgb_h)

                    # Scale the coordinates to the depth image resolution
                    depth_cx = int(rgb_cx * scale_x)
                    depth_cy = int(rgb_cy * scale_y)

                    win_size = 10 # This should match the win_size in get_marker_pose
                    x_start_depth_rect = max(0, depth_cx - win_size // 2)
                    y_start_depth_rect = max(0, depth_cy - win_size // 2)
                    x_end_depth_rect = min(depth_w, depth_cx + win_size // 2)
                    y_end_depth_rect = min(depth_h, depth_cy + win_size // 2)

                    # Normalize depth image for visualization (0-255)
                    depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR) # Convert to BGR for colored rectangle

                    cv2.rectangle(depth_display, (x_start_depth_rect, y_start_depth_rect), (x_end_depth_rect, y_end_depth_rect), (0, 255, 255), 2) # Yellow rectangle
                    self.debug_depth_image = depth_display
                else:
                    # If no marker is detected, still show the depth image (optional)
                    depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)
                    self.debug_depth_image = depth_display

                # Log depth image info for debugging
                rospy.loginfo_throttle(5.0, f"Depth Image Dimensions: {depth_image.shape[1]}x{depth_image.shape[0]}")
                if depth_image.shape[0] > 0 and depth_image.shape[1] > 0:
                    sample_depth_value = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]
                    rospy.loginfo_throttle(5.0, f"Sample Depth Value (center): {sample_depth_value:.2f}")

        except Exception as e:
            rospy.logerr(f"An error occurred in image_callback: {e}")

    def get_marker_pose(self, corners, rgb_image_shape, depth_image):
        # Get image dimensions for scaling
        rgb_h, rgb_w, _ = rgb_image_shape
        depth_h, depth_w = depth_image.shape

        # Calculate scaling factors
        scale_x = depth_w / float(rgb_w)
        scale_y = depth_h / float(rgb_h)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(np.array([corners]), 0.15, self.cam_matrix, self.dist_coeffs)
        rvec = rvecs[0][0]
        
        # Get the center of the marker in RGB pixel coordinates
        rgb_cx = int(np.mean(corners[:, 0]))
        rgb_cy = int(np.mean(corners[:, 1]))

        # Scale the coordinates to the depth image resolution
        depth_cx = int(rgb_cx * scale_x)
        depth_cy = int(rgb_cy * scale_y)

        # Get a more robust distance from the depth image by taking a median of a central region
        win_size = 10
        x_start = max(0, depth_cx - win_size // 2)
        x_end = min(depth_w, depth_cx + win_size // 2)
        y_start = max(0, depth_cy - win_size // 2)
        y_end = min(depth_h, depth_cy + win_size // 2)
        
        depth_region = depth_image[y_start:y_end, x_start:x_end]
        rospy.loginfo_throttle(1.0, f"  [DEBUG] Raw depth region values:\n{depth_region}")
        
        valid_depths = depth_region[np.isfinite(depth_region) & (depth_region > 0)]

        if len(valid_depths) == 0:
            rospy.logwarn_throttle(1.0, f"No valid depth data in the central region of the marker at RGB({rgb_cx}, {rgb_cy}) -> Depth({depth_cx}, {depth_cy}). Ignoring.")
            rospy.loginfo_throttle(1.0, f"  [DEBUG] isfinite mask:\n{np.isfinite(depth_region)}")
            rospy.loginfo_throttle(1.0, f"  [DEBUG] greater than zero mask:\n{(depth_region > 0)}")
            rospy.loginfo_throttle(1.0, f"  [DEBUG] Combined mask:\n{np.isfinite(depth_region) & (depth_region > 0)}")
            rospy.loginfo_throttle(1.0, f"  [DEBUG] Valid depths found: {valid_depths}")
            return None, None
        
        distance_z = np.median(valid_depths)
        rospy.loginfo_throttle(1.0, f"  [DEBUG] Calculated median distance_z: {distance_z:.4f}")

        if distance_z <= 0:
            rospy.logwarn_throttle(1.0, f"Median depth value for marker is invalid ({distance_z:.2f}). Ignoring.")
            return None, None
        
        # Calculate the accurate 3D position in the camera frame
        # IMPORTANT: Use the original RGB coordinates for reprojection to maintain geometric accuracy with the camera intrinsics
        x = (rgb_cx - self.center_x) * distance_z / self.focal_x
        y = (rgb_cy - self.center_y) * distance_z / self.focal_y
        tvec = np.array([x, y, distance_z])
        
        return rvec, tvec

    def transform_to_odom(self, rvec, tvec, stamp):
        try:
            # Create a PoseStamped message for the marker in the camera frame
            marker_pose_camera = PoseStamped()
            marker_pose_camera.header.stamp = stamp
            marker_pose_camera.header.frame_id = self.camera_frame
            marker_pose_camera.pose.position.x = tvec[0]
            marker_pose_camera.pose.position.y = tvec[1]
            marker_pose_camera.pose.position.z = tvec[2]
            
            R, _ = cv2.Rodrigues(rvec)
            T = np.identity(4)
            T[:3, :3] = R
            q = quaternion_from_matrix(T)
            marker_pose_camera.pose.orientation.x = q[0]
            marker_pose_camera.pose.orientation.y = q[1]
            marker_pose_camera.pose.orientation.z = q[2]
            marker_pose_camera.pose.orientation.w = q[3]

            # Transform the marker pose to the odom frame
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.camera_frame, stamp, rospy.Duration(1.0))
            marker_pose_odom = tf2_geometry_msgs.do_transform_pose(marker_pose_camera, transform)
            return marker_pose_odom
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not transform from {self.camera_frame} to {self.target_frame}: {e}")
            return None

    def run(self):
        rate = rospy.Rate(60) # 60hz
        while not rospy.is_shutdown():
            if self.enable_visualization:
                if self.debug_image is not None:
                    cv2.imshow("Aruco Goal Debug", self.debug_image)
                if self.debug_depth_image is not None:
                    cv2.imshow("Aruco Depth Debug", self.debug_depth_image)
                cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ArucoGoalPublisher()
        if not rospy.is_shutdown():
            node.run()
    except rospy.ROSInterruptException:
        pass