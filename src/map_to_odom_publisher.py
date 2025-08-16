#!/usr/bin/env python

import rospy
import tf2_ros
import tf_conversions
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

class MapToOdomPublisher:
    def __init__(self):
        rospy.init_node('map_to_odom_publisher', anonymous=True)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.global_pos_sub = rospy.Subscriber(
            '/mavros/global_position/local', Odometry, self.global_pos_callback
        )
        self.local_odom_sub = rospy.Subscriber(
            '/mavros/local_position/odom', Odometry, self.local_odom_callback
        )

        self.map_T_base_link = None
        self.odom_T_base_link = None

        rospy.loginfo("Map to Odom Publisher Node Started.")

    def global_pos_callback(self, msg):
        # This message provides map -> base_link
        self.map_T_base_link = msg.pose.pose

    def local_odom_callback(self, msg):
        # This message provides odom -> base_link
        self.odom_T_base_link = msg.pose.pose
        self.publish_map_to_odom_tf()

    def publish_map_to_odom_tf(self):
        if self.map_T_base_link is None or self.odom_T_base_link is None:
            return

        # Get map_T_base_link (from global_position/local)
        map_pos = self.map_T_base_link.position
        map_ori = self.map_T_base_link.orientation

        # Get odom_T_base_link (from local_position/odom)
        odom_pos = self.odom_T_base_link.position
        odom_ori = self.odom_T_base_link.orientation

        # Convert to TF transforms
        map_T_base_link_tf = tf_conversions.transformations.translation_matrix([map_pos.x, map_pos.y, map_pos.z])
        map_T_base_link_tf = map_T_base_link_tf @ tf_conversions.transformations.quaternion_matrix([map_ori.x, map_ori.y, map_ori.z, map_ori.w])

        odom_T_base_link_tf = tf_conversions.transformations.translation_matrix([odom_pos.x, odom_pos.y, odom_pos.z])
        odom_T_base_link_tf = odom_T_base_link_tf @ tf_conversions.transformations.quaternion_matrix([odom_ori.x, odom_ori.y, odom_ori.z, odom_ori.w])

        # Compute base_link_T_odom (inverse of odom_T_base_link)
        base_link_T_odom_tf = tf_conversions.transformations.inverse_matrix(odom_T_base_link_tf)

        # Compute map_T_odom = map_T_base_link * base_link_T_odom
        map_T_odom_tf = map_T_base_link_tf @ base_link_T_odom_tf

        # Extract translation and rotation for the TransformStamped message
        translation = tf_conversions.transformations.translation_from_matrix(map_T_odom_tf)
        rotation = tf_conversions.transformations.quaternion_from_matrix(map_T_odom_tf)

        # Create and broadcast the TransformStamped message
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]

        self.tf_broadcaster.sendTransform(t)

if __name__ == '__main__':
    try:
        node = MapToOdomPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
