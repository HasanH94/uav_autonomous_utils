#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_publisher():
    """
    This node publishes a custom CameraInfo message with pre-defined calibration values.
    """
    rospy.init_node('custom_camera_info_publisher', anonymous=True)
    pub = rospy.Publisher('/calibrated_camera_info', CameraInfo, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    # Create a CameraInfo message and populate it with your calibration data
    cam_info = CameraInfo()

    cam_info.header.frame_id = "camera_link" # Or your relevant frame
    cam_info.height = 480
    cam_info.width = 848
    cam_info.distortion_model = "plumb_bob"

    # Distortion coefficients
    cam_info.D = [-0.0008133802375829636, 0.000718938482982905, 4.129454172570969e-05, 7.253403764114478e-05, 0.0]

    # Intrinsic camera matrix (K)
    cam_info.K = [454.5709225684875, 0.0, 423.20333549657533,
                  0.0, 454.55174808970844, 239.58682770178422,
                  0.0, 0.0, 1.0]

    # Rectification matrix (R)
    cam_info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]

    # Projection/camera matrix (P)
    cam_info.P = [454.5959599919915, 0.0, 423.2985880012217, 0.0,
                  0.0, 454.61877650819685, 239.61869197441868, 0.0,
                  0.0, 0.0, 1.0, 0.0]

    rospy.loginfo("Publishing custom camera info on /calibrated_camera_info")

    while not rospy.is_shutdown():
        cam_info.header.stamp = rospy.Time.now()
        pub.publish(cam_info)
        rate.sleep()

if __name__ == '__main__':
    try:
        camera_info_publisher()
    except rospy.ROSInterruptException:
        pass
