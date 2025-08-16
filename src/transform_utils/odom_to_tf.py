#!/usr/bin/env python
import rospy
import tf
from nav_msgs.msg import Odometry

def odom_callback(msg):
    br = tf.TransformBroadcaster()
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    br.sendTransform((pos.x, pos.y, pos.z),
                     (ori.x, ori.y, ori.z, ori.w),
                     rospy.Time.now(),
                     "base_link",
                     "odom")

if __name__ == '__main__':
    rospy.init_node('odom_to_tf_broadcaster')
    rospy.Subscriber('/mavros/local_position/odom', Odometry, odom_callback)
    rospy.spin()
