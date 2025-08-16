#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg

def main():
    rospy.init_node('tf_to_pose_publisher')
    topic = rospy.get_param('~topic', '/camera/pose')
    pub = rospy.Publisher(topic, geometry_msgs.msg.PoseStamped, queue_size=1)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rate = rospy.Rate(50.0)

    parent = rospy.get_param('~parent_frame', 'odom')
    child = rospy.get_param('~child_frame', 'camera_link')
    rospy.loginfo("Publishing PoseStamped from '%s' to '%s'", parent, child)

    while not rospy.is_shutdown():
        try:
            trans = tf_buffer.lookup_transform(parent, child,
                                               rospy.Time.now(),
                                               rospy.Duration(1.0))
            msg = geometry_msgs.msg.PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = parent
            msg.pose.position = trans.transform.translation
            msg.pose.orientation = trans.transform.rotation
            pub.publish(msg)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform error: %s", e)

        rate.sleep()

if __name__ == '__main__':
    main()
