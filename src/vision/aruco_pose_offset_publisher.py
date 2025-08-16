#!/usr/bin/env python3

import rospy
import cv2
from cv2 import aruco
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose # Import Pose for individual poses
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from tf.transformations import quaternion_matrix, quaternion_from_matrix, translation_matrix, concatenate_matrices, rotation_matrix
import message_filters # For TimeSynchronizer if needed, though not directly used in this version

class ArucoPoseOffsetPublisher:
    def __init__(self):
        rospy.init_node('aruco_pose_offset_publisher', anonymous=True)

        # --- Parameters ---
        # self.marker_id = rospy.get_param('~marker_id', -1) # No longer filtering by specific ID
        self.marker_size = rospy.get_param('~marker_size', 1.2) # Physical size of the marker in meters
        self.z_offset = rospy.get_param('~z_offset', 1.0) # Desired offset along marker's local Z-axis
        self.y_offset = rospy.get_param('~y_offset', 0.0) # Desired offset along marker's local Y-axis
        self.x_offset = rospy.get_param('~x_offset', 0.0) # Desired offset along marker's local X-axis
        self.camera_rgb_topic = rospy.get_param('~camera_rgb_topic', '/iris/camera/rgb/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/iris/camera/rgb/camera_info')
        self.output_goal_topic = rospy.get_param('~output_goal_topic', '/aruco_offset_pose') # Output topic for the selected goal
        self.target_frame = rospy.get_param('~target_frame', 'odom') # Frame to publish the final pose in
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        self.min_marker_pixel_area = rospy.get_param('~min_marker_pixel_area', 400) # Minimum pixel area for a valid detection
        self.max_marker_pixel_area = rospy.get_param('~max_marker_pixel_area', 100000) # Maximum pixel area for a valid detection

        # --- ROS Components ---
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Aruco Setup ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        # --- Camera Intrinsics (will be populated from CameraInfo) ---
        self.cam_matrix = None
        self.dist_coeffs = None
        self.camera_frame_id = None
        self.camera_info_received = False

        # --- State Variables ---
        self.mission_goal = None # Stores the latest mission goal from /move_base_mission/goal

        # --- Publishers ---
        self.selected_offset_pose_pub = rospy.Publisher(self.output_goal_topic, PoseStamped, queue_size=1)
        self.detection_status_pub = rospy.Publisher('/aruco_detection_status', Bool, queue_size=1)

        # --- Subscribers ---
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.camera_rgb_topic, Image, self.image_callback)
        rospy.Subscriber('/move_base_mission', PoseStamped, self.mission_goal_callback) # Subscribe to mission goal

        rospy.loginfo("ArucoPoseOffsetPublisher node initialized. Waiting for camera info...")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.cam_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.camera_frame_id = msg.header.frame_id
            self.camera_info_received = True
            rospy.loginfo(f"Received camera info for frame: {self.camera_frame_id}")

    def mission_goal_callback(self, msg):
        self.mission_goal = msg
        rospy.loginfo_throttle(5.0, f"Received mission goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}, z={msg.pose.position.z:.2f}")

    def image_callback(self, img_msg):
        if not self.camera_info_received:
            rospy.logwarn_throttle(5, "Waiting for camera info before processing images.")
            self.detection_status_pub.publish(False)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            self.detection_status_pub.publish(False)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        marker_detected_this_frame = False
        valid_markers_data = [] # To store {id, corners, rvec, tvec, offset_pose_stamped_odom, area}

        if ids is not None:
            for i in range(len(ids)): # Loop through all detected markers
                current_marker_id = ids[i][0]

                # Filter by target_marker_id if specified
                # if self.target_marker_id != -1 and current_marker_id != self.target_marker_id:
                #     continue

                # Calculate pixel area for filtering
                marker_corners_pixels = corners[i][0]
                area = cv2.contourArea(marker_corners_pixels)

                if not (self.min_marker_pixel_area <= area <= self.max_marker_pixel_area):
                    rospy.logwarn_throttle(1.0, f"Marker ID {current_marker_id} detected but area {area:.0f} is outside valid range [{self.min_marker_pixel_area}, {self.max_marker_pixel_area}]. Skipping.")
                    continue

                # If we reach here, the marker passed the area filter, so it's a valid detection for this frame

                # Estimate pose of the current marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], self.marker_size, self.cam_matrix, self.dist_coeffs
                )
                # Extract the single rvec and tvec
                rvec = rvec[0][0]
                tvec = tvec[0][0]

                # --- Get Marker's Rotation Matrix in Camera Frame ---
                R_marker_in_camera, _ = cv2.Rodrigues(rvec)

                # --- Ambiguity Check & Pose Correction ---
                # A correctly detected marker should have its Z-axis pointing roughly towards the camera.
                # The vector from the camera to the marker is tvec. So, the dot product of
                # the marker's Z-axis and tvec should be negative. If it's positive, the pose is flipped.
                if np.dot(R_marker_in_camera[:, 2], tvec) > 0:
                    rospy.logwarn_throttle(1.0, f"Marker ID {current_marker_id} pose is ambiguous. Correcting orientation.")
                    # Correct the pose by rotating it 180 degrees around the marker's local Y-axis.
                    # This operation flips the directions of the marker's X and Z axes.
                    R_marker_in_camera[:, 0] = -R_marker_in_camera[:, 0]
                    R_marker_in_camera[:, 2] = -R_marker_in_camera[:, 2]
                    # The rvec must be updated to reflect this change for consistent visualization.
                    rvec, _ = cv2.Rodrigues(R_marker_in_camera)

                # --- Construct Desired Rotation Matrix for Goal Pose (Drone faces marker, level) ---
                # The goal is to have the drone's front (X-axis) point directly towards the marker,
                # while keeping the drone level with respect to the marker's plane.
                # This is achieved by setting the goal axes relative to the marker's axes:
                # Goal's Forward (X) = Marker's Inward (-Z)
                # Goal's Up (Z)       = Marker's Up (-Y)
                # Goal's Left (Y)     = Marker's Right (X)
                # This defines a standard "facing" orientation, assuming a drone body frame of (X-fwd, Y-left, Z-up)
                # and a marker frame of (X-right, Y-down, Z-out).
                # Columns of R_marker_in_camera are [X_marker_axis, Y_marker_axis, Z_marker_axis]
                X_marker_axis = R_marker_in_camera[:, 0]
                Y_marker_axis = R_marker_in_camera[:, 1]
                Z_marker_axis = R_marker_in_camera[:, 2]

                R_goal_in_camera = np.array([
                    -Z_marker_axis,  # Goal X: Aligns with Marker's -Z axis (points inward)
                    X_marker_axis,   # Goal Y: Aligns with Marker's  X axis (points left, relative to drone)
                    -Y_marker_axis   # Goal Z: Aligns with Marker's -Y axis (points up, relative to marker)
                ]).T # Transpose to get columns as axes

                # --- Calculate Final Translation Vector for Goal Pose in Camera Frame ---
                # After the ambiguity check, the marker's Z-axis is guaranteed to point towards the camera.
                # To place the goal at a consistent offset, we apply the offset along this reliable axis.
                # Subtracting the scaled Z-axis vector moves the goal away from the camera, "behind" the marker.
                marker_z_axis_in_camera = R_marker_in_camera[:, 2]
                marker_y_axis_in_camera = R_marker_in_camera[:, 1]
                marker_x_axis_in_camera = R_marker_in_camera[:, 0]
                tvec_goal = tvec + marker_z_axis_in_camera * self.z_offset + marker_y_axis_in_camera * self.y_offset + marker_x_axis_in_camera * self.x_offset

                # --- Construct Full Homogeneous Transformation Matrix for Goal Pose in Camera Frame ---
                T_goal_in_camera = np.eye(4)
                T_goal_in_camera[:3, :3] = R_goal_in_camera
                T_goal_in_camera[:3, 3] = tvec_goal

                # Extract rvec and tvec from this final matrix (for visualization and PoseStamped)
                rvec_final, _ = cv2.Rodrigues(T_goal_in_camera[:3, :3])
                tvec_final = T_goal_in_camera[:3, 3]

                # --- Transform to Target Frame (odom) ---
                final_pose_camera = PoseStamped()
                final_pose_camera.header.stamp = img_msg.header.stamp
                final_pose_camera.header.frame_id = self.camera_frame_id
                final_pose_camera.pose.position.x = tvec_final[0]
                final_pose_camera.pose.position.y = tvec_final[1]
                final_pose_camera.pose.position.z = tvec_final[2]

                q_final = quaternion_from_matrix(T_goal_in_camera)
                final_pose_camera.pose.orientation.x = q_final[0]
                final_pose_camera.pose.orientation.y = q_final[1]
                final_pose_camera.pose.orientation.z = q_final[2]
                final_pose_camera.pose.orientation.w = q_final[3]

                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.target_frame, final_pose_camera.header.frame_id, 
                        img_msg.header.stamp, rospy.Duration(1.0)
                    )
                    final_offset_pose_stamped_odom = tf2_geometry_msgs.do_transform_pose(final_pose_camera, transform)
                    
                    valid_markers_data.append({
                        'id': current_marker_id,
                        'corners': corners[i],
                        'rvec': rvec,
                        'tvec': tvec,
                        'rvec_final': rvec_final,
                        'tvec_final': tvec_final,
                        'offset_pose_stamped_odom': final_offset_pose_stamped_odom,
                        'area': area
                    })

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn_throttle(1.0, f"Could not transform offset pose for Marker ID {current_marker_id} from {final_pose_camera.header.frame_id} to {self.target_frame}: {e}")
                    # This marker will not be considered for selection

        # --- Marker Selection Logic ---
        best_marker_data = None
        if valid_markers_data:
            if self.mission_goal is not None:
                # Find closest to mission goal
                min_dist = float('inf')
                for marker_data in valid_markers_data:
                    pose_odom = marker_data['offset_pose_stamped_odom'].pose.position
                    goal_pos = self.mission_goal.pose.position
                    dist = np.linalg.norm([pose_odom.x - goal_pos.x, pose_odom.y - goal_pos.y, pose_odom.z - goal_pos.z])
                    if dist < min_dist:
                        min_dist = dist
                        best_marker_data = marker_data
                rospy.loginfo_throttle(1.0, f"Selected Marker ID {best_marker_data['id']} (closest to mission goal, dist: {min_dist:.2f}m).")
            else:
                # No mission goal, pick largest area
                max_area = -1
                for marker_data in valid_markers_data:
                    if marker_data['area'] > max_area:
                        max_area = marker_data['area']
                        best_marker_data = marker_data
                rospy.logwarn_throttle(1.0, f"No mission goal. Selected Marker ID {best_marker_data['id']} (largest area: {max_area:.0f}).")

        # --- Publish Selected Pose ---
        if best_marker_data:
            self.selected_offset_pose_pub.publish(best_marker_data['offset_pose_stamped_odom'])
            rospy.loginfo_throttle(1.0, f"Published selected offset pose for Marker ID {best_marker_data['id']}.")

        if valid_markers_data:
            marker_detected_this_frame = True

        # Determine final detection status based on valid markers
        final_detection_status = Bool()
        final_detection_status.data = bool(valid_markers_data) # True if list is not empty, False otherwise
        self.detection_status_pub.publish(final_detection_status)

        # --- Visualization (Optional) ---
        if self.enable_visualization:
            if ids is not None:
                # Draw all detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

                for marker_data in valid_markers_data:
                    # Draw axes for the original detected marker
                    cv2.drawFrameAxes(cv_image, self.cam_matrix, self.dist_coeffs, marker_data['rvec'], marker_data['tvec'], self.marker_size * 0.5)
                    # Draw axes for the final offset pose (in camera frame)
                    cv2.drawFrameAxes(cv_image, self.cam_matrix, self.dist_coeffs, marker_data['rvec_final'], marker_data['tvec_final'], self.marker_size * 0.7, 3)

                    # Display info for each marker
                    text_color = (0, 255, 0) # Green for non-selected
                    if best_marker_data and marker_data['id'] == best_marker_data['id']:
                        text_color = (0, 0, 255) # Red for selected

                    cv2.putText(cv_image, f"ID: {marker_data['id']}", (int(marker_data['corners'][0][0][0]), int(marker_data['corners'][0][0][1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    cv2.putText(cv_image, f"Area: {marker_data['area']:.0f}", (int(marker_data['corners'][0][0][0]), int(marker_data['corners'][0][0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            cv2.imshow("Aruco Poses with Offset", cv_image)
            cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ArucoPoseOffsetPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
