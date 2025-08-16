import rospy
import tf2_ros
import tf
import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from trajectory_msgs.msg import MultiDOFJointTrajectory
from std_srvs.srv import Empty, EmptyResponse

class VisualServoingNode:
    def __init__(self):
        self.nh = rospy.NodeHandle()
        self.private_nh = rospy.get_param('~')
        self.tf2 = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf2)

        self.received_odometry = False
        self.received_pole_pose = False
        self.activated = False
        self.states = []

        self.initialize_subscribers()
        self.initialize_publishers()
        self.initialize_services()
        self.load_params()
        self.load_tfs()

        self.timer_run = rospy.Timer(rospy.Duration(1.0 / 200.0), self.run)
    
    def initialize_subscribers(self):
        self.odometry_sub = rospy.Subscriber("/odometry", Odometry, self.odometry_callback)
        self.pole_vicon_sub = rospy.Subscriber("geranos_pole_white/vrpn_client/estimated_transform", TransformStamped, self.pole_vicon_callback)
        self.pose_estimate_sub = rospy.Subscriber("PolePoseNode/EstimatedPose", PoseStamped, self.pose_estimate_callback)
    
    def initialize_publishers(self):
        self.pub_trajectory = rospy.Publisher("/command_trajectory", MultiDOFJointTrajectory, queue_size=0)
        self.pole_pos_pub = rospy.Publisher(rospy.get_name() + "/estimated_pole_position", PointStamped, queue_size=0)
        self.pub_markers = rospy.Publisher(rospy.get_name() + "/trajectory_markers", MarkerArray, queue_size=0)
        self.error_pub = rospy.Publisher(rospy.get_name() + "/error_vector", PointStamped, queue_size=0)
        self.transformed_odom_pub = rospy.Publisher(rospy.get_name() + "/transformed_odometry", Odometry, queue_size=1)
        self.yaw_pub = rospy.Publisher(rospy.get_name() + "/yaw", PointStamped, queue_size=0)
    
    def initialize_services(self):
        self.activate_service = rospy.Service("activate_servoing_service", Empty, self.activate_servoing_srv)
        self.grab_pole_service = rospy.Service("grab_pole_service", Empty, self.grab_pole_srv)
        self.lift_pole_service = rospy.Service("lift_pole_service", Empty, self.lift_pole_srv)
    
    def odometry_callback(self, odometry_msg):
        rospy.loginfo_once("VisualServoingNode received first odometry!")
        odom = self.msg_to_eigen_odometry(odometry_msg)
        self.transform_odometry(odom)
        self.publish_odometry(odom)
        self.current_odometry = odom
        self.received_odometry = True

    def publish_odometry(self, odom):
        odometry_msg = self.eigen_to_msg_odometry(odom)
        self.transformed_odom_pub.publish(odometry_msg)
    
    def transform_odometry(self, odom):
        R_B_imu = self.T_B_imu.rotation
        r_B_imu_B = self.T_B_imu.translation
        R_W_B = odom["orientation_W_B"].rotation_matrix()

        odom["position_W"] -= R_W_B.dot(R_B_imu.dot(r_B_imu_B))
        v_rot = np.cross(odom["angular_velocity_B"], r_B_imu_B)
        odom["velocity_B"] = R_B_imu.dot(odom["velocity_B"]) - v_rot

    def pose_estimate_callback(self, pose_msg):
        pole_pos_C = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        pole_pos_B = self.t_B_cam + self.R_B_cam.dot(pole_pos_C)
        self.current_pole_pos_B = pole_pos_B

        R_W_B = self.current_odometry["orientation_W_B"].rotation_matrix()
        self.current_pole_pos = self.current_odometry["position_W"] + R_W_B.dot(pole_pos_B)
        self.received_pole_pose = True

        pole_pos_msg = self.eigen_to_point_msg(self.current_pole_pos)
        self.pole_pos_pub.publish(pole_pos_msg)

    def pole_vicon_callback(self, pole_transform_msg):
        rospy.loginfo_once("[VisualServoingNode] Received first transform of Pole!")
        self.pole_trajectory_point = self.transform_msg_to_eigen(pole_transform_msg)
        self.current_pole_pos_vicon = self.pole_trajectory_point["position_W"]

        if self.received_pole_pose:
            error_vector = self.current_pole_pos_vicon - self.current_pole_pos
            error_msg = self.eigen_to_point_msg(error_vector)
            self.error_pub.publish(error_msg)
    
    def grab_pole_srv(self, request, response):
        if self.activated:
            self.activated = False
        
        waypoint_position = self.current_odometry["position_W"] - np.array([0.0, 0.0, 1.1])
        waypoint_orientation = self.quaternion_from_yaw(self.start_yaw)
        velocity_command = np.array([0.0, 0.0, 0.0])
        ang_velocity_command = np.array([0.0, 0.0, 0.0])
        duration = 5.0

        trajectory_msg = self.generate_trajectory_msg(waypoint_position, waypoint_orientation, velocity_command, ang_velocity_command, duration)
        self.pub_trajectory.publish(trajectory_msg)

        markers = self.create_rviz_markers()
        self.pub_markers.publish(markers)
        return EmptyResponse()

    def lift_pole_srv(self, request, response):
        if self.activated:
            self.activated = False

        waypoint_position = self.current_odometry["position_W"] + np.array([0.0, 0.0, 1.1])
        waypoint_orientation = self.quaternion_from_yaw(self.start_yaw)
        velocity_command = np.array([0.0, 0.0, 0.0])
        ang_velocity_command = np.array([0.0, 0.0, 0.0])
        duration = 5.0

        trajectory_msg = self.generate_trajectory_msg(waypoint_position, waypoint_orientation, velocity_command, ang_velocity_command, duration)
        self.pub_trajectory.publish(trajectory_msg)

        markers = self.create_rviz_markers()
        self.pub_markers.publish(markers)
        return EmptyResponse()
    
    def load_params(self):
        self.max_v = rospy.get_param(rospy.get_name() + "/max_v", 1.0)
        self.max_a = rospy.get_param(rospy.get_name() + "/max_a", 1.0)
        self.max_ang_v = rospy.get_param(rospy.get_name() + "/max_ang_v", 1.0)
        self.max_ang_a = rospy.get_param(rospy.get_name() + "/max_ang_a", 1.0)
        self.sampling_time = rospy.get_param(rospy.get_name() + "/sampling_time", 0.01)
        self.k_p = rospy.get_param(rospy.get_name() + "/k_p", 1.0)
        self.k_p_ang = rospy.get_param(rospy.get_name() + "/k_p_ang", 1.0)

    def load_tfs(self):
        rospy.loginfo_once("[VisualServoingNode] loading TFs")
        try:
            self.tf_listener.waitForTransform("base", "imu", rospy.Time(0), rospy.Duration(5.0))
            tf_base_imu = self.tf_listener.lookupTransform("base", "imu", rospy.Time(0))
            self.T_B_imu = self.transform_to_eigen(tf_base_imu)
            rospy.loginfo("[VisualServoingNode] Found base to imu transform!")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as ex:
            rospy.logerr("[VisualServoingNode] %s", str(ex))

        try:
            self.tf_listener.waitForTransform("base", "cam", rospy.Time(0), rospy.Duration(5.0))
            tf_base_cam = self.tf_listener.lookupTransform("base", "cam", rospy.Time(0))
            self.T_B_cam = self.transform_to_eigen(tf_base_cam)
            self.R_B_cam = self.T_B_cam.rotation()
            self.t_B_cam = self.T_B_cam.translation()
            rospy.loginfo("[VisualServoingNode] Found base to cam transform!")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as ex:
            rospy.logerr("[VisualServoingNode] %s", str(ex))

    def activate_servoing_srv(self, request, response):
        self.start_yaw = self.yaw_from_quaternion(self.current_odometry["orientation_W_B"])
        self.start_position = self.current_odometry["position_W"]
        self.velocity_integral = np.zeros(3)
        self.activated = True
        return EmptyResponse()
    
    def run(self, event):
        if not self.received_odometry or not self.received_pole_pose or not self.activated:
            return
        
        R_W_B = self.current_odometry["orientation_W_B"].rotation_matrix()
        r_B = self.current_pole_pos_B - np.array([0.0, 0.0, 1.1])
        waypoint_pos = self.current_odometry["position_W"] + R_W_B.dot(r_B)

        velocity_command = self.k_p * (waypoint_pos - self.current_odometry["position_W"])
        if np.linalg.norm(velocity_command) > self.max_v:
            velocity_command = velocity_command * (self.max_v / np.linalg.norm(velocity_command))

        velocity_msg = self.eigen_to_twist_msg(velocity_command)
        self.pub_trajectory.publish(velocity_msg)

if __name__ == "__main__":
    rospy.init_node('visual_servoing_node')
    vs_node = VisualServoingNode()
    rospy.spin()
