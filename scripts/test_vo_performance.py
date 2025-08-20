#!/usr/bin/env python3
"""
Test script to measure VO filter performance.
Publishes test point clouds and measures processing rate.
"""

import rospy
import numpy as np
import time
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header

class VOPerformanceTester:
    def __init__(self):
        rospy.init_node('vo_performance_tester')
        
        # Publishers
        self.cloud_pub = rospy.Publisher('/iris/camera/depth/points', PointCloud2, queue_size=1)
        self.vel_pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=1)
        
        # Subscriber to measure output rate
        self.output_count = 0
        self.last_print_time = time.time()
        rospy.Subscriber('/mavros/setpoint_velocity/cmd_vel', TwistStamped, self.output_cb)
        
        rospy.loginfo("Performance tester ready. Publishing test data...")
        
    def output_cb(self, msg):
        self.output_count += 1
        
        # Print rate every second
        now = time.time()
        if now - self.last_print_time >= 1.0:
            rate = self.output_count / (now - self.last_print_time)
            rospy.loginfo(f"VO Filter output rate: {rate:.1f} Hz")
            self.output_count = 0
            self.last_print_time = now
    
    def generate_test_cloud(self, n_points=1000):
        """Generate a test point cloud with obstacles."""
        # Generate random obstacles in front of drone
        points = []
        
        # Add some nearby obstacles
        for _ in range(n_points):
            # Random points in a cone in front
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi/4)  # 45 degree cone
            r = np.random.uniform(0.5, 5.0)
            
            x = r * np.sin(phi) * np.cos(theta)  # forward
            y = r * np.sin(phi) * np.sin(theta)  # left/right
            z = r * np.cos(phi) - 1.0  # up/down (offset down)
            
            points.append([x, y, z])
        
        # Create PointCloud2 message
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # Camera frame
        
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        return cloud_msg
    
    def run(self):
        rate = rospy.Rate(30)  # Publish at 30 Hz
        
        while not rospy.is_shutdown():
            # Publish test point cloud
            cloud = self.generate_test_cloud(n_points=2000)
            self.cloud_pub.publish(cloud)
            
            # Publish test velocity command
            vel = TwistStamped()
            vel.header.stamp = rospy.Time.now()
            vel.header.frame_id = "odom"
            vel.twist.linear.x = 1.0  # Forward at 1 m/s
            vel.twist.linear.y = 0.0
            vel.twist.linear.z = 0.0
            vel.twist.angular.z = 0.0
            self.vel_pub.publish(vel)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        tester = VOPerformanceTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass