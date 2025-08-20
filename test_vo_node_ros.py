#!/usr/bin/env python3
"""
ROS-based test for the actual VO filter node.
This tests the real node with actual ROS messages.
"""

import rospy
import numpy as np
import time
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import struct

class VONodeROSTester:
    def __init__(self):
        rospy.init_node('vo_filter_tester')
        
        # Publishers
        self.vel_pub = rospy.Publisher('/uav/attractive_velocity', TwistStamped, queue_size=1)
        self.cloud_pub = rospy.Publisher('/iris/camera/depth/points', PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher('/mavros/local_position/odom', Odometry, queue_size=1)
        
        # Subscriber to catch output
        self.output_velocity = None
        self.output_received = False
        self.output_count = 0
        self.last_output_time = None
        rospy.Subscriber('/mavros/setpoint_velocity/cmd_vel', TwistStamped, self.output_cb)
        
        # Test results storage
        self.test_results = []
        
        # Wait for node to be ready
        rospy.loginfo("Waiting for VO filter node to be ready...")
        rospy.sleep(2.0)

    def output_cb(self, msg):
        """Callback to receive output from VO filter"""
        self.output_velocity = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])
        self.output_received = True
        self.output_count += 1
        self.last_output_time = rospy.Time.now()

    def create_point_cloud(self, points):
        """Create PointCloud2 message from numpy array of points"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"  # Camera frame
        
        # Create PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        cloud = point_cloud2.create_cloud(header, fields, points)
        return cloud

    def publish_odometry(self, position=[0,0,2], orientation=[0,0,0,1]):
        """Publish drone odometry"""
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        # Set position
        odom.pose.pose.position.x = position[0]
        odom.pose.pose.position.y = position[1]
        odom.pose.pose.position.z = position[2]
        
        # Set orientation (quaternion)
        odom.pose.pose.orientation.x = orientation[0]
        odom.pose.pose.orientation.y = orientation[1]
        odom.pose.pose.orientation.z = orientation[2]
        odom.pose.pose.orientation.w = orientation[3]
        
        self.odom_pub.publish(odom)

    def publish_desired_velocity(self, velocity):
        """Publish desired velocity"""
        twist = TwistStamped()
        twist.header.stamp = rospy.Time.now()
        twist.header.frame_id = "odom"
        twist.twist.linear.x = velocity[0]
        twist.twist.linear.y = velocity[1]
        twist.twist.linear.z = velocity[2]
        twist.twist.angular.z = 0.0
        
        self.vel_pub.publish(twist)

    def wait_for_output(self, timeout=1.0):
        """Wait for output from VO filter"""
        start_time = rospy.Time.now()
        rate = rospy.Rate(100)  # 100 Hz checking
        
        self.output_received = False
        while not rospy.is_shutdown():
            if self.output_received:
                return True
            
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                return False
            
            rate.sleep()
        
        return False

    def run_test_case(self, name, description, desired_velocity, obstacle_points, expected_check):
        """Run a single test case"""
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo(f"Test: {name}")
        rospy.loginfo(f"Description: {description}")
        
        # Reset output tracking
        self.output_velocity = None
        self.output_received = False
        
        # Publish odometry (drone at origin, facing forward)
        self.publish_odometry([0, 0, 2], [0, 0, 0, 1])
        rospy.sleep(0.1)
        
        # Publish point cloud (obstacles)
        if len(obstacle_points) > 0:
            cloud = self.create_point_cloud(obstacle_points)
            self.cloud_pub.publish(cloud)
        rospy.sleep(0.1)
        
        # Publish desired velocity
        self.publish_desired_velocity(desired_velocity)
        
        # Wait for output
        if self.wait_for_output(timeout=1.0):
            # Check if output is as expected
            input_speed = np.linalg.norm(desired_velocity)
            output_speed = np.linalg.norm(self.output_velocity)
            
            passed = expected_check(desired_velocity, self.output_velocity)
            
            rospy.loginfo(f"Input:  {desired_velocity} ({input_speed:.2f} m/s)")
            rospy.loginfo(f"Output: {self.output_velocity} ({output_speed:.2f} m/s)")
            rospy.loginfo(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
            
            self.test_results.append({
                'name': name,
                'passed': passed,
                'input': desired_velocity,
                'output': self.output_velocity
            })
            
            return passed
        else:
            rospy.logwarn("Timeout waiting for output!")
            self.test_results.append({
                'name': name,
                'passed': False,
                'error': 'Timeout'
            })
            return False

    def run_performance_test(self, duration=10.0):
        """Test processing rate over time"""
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo("PERFORMANCE TEST")
        rospy.loginfo(f"Running for {duration} seconds...")
        
        # Generate a complex scenario
        obstacle_points = []
        for i in range(100):  # 100 obstacles
            x = np.random.uniform(1, 5)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(-1, 1)
            obstacle_points.append([x, y, z])
        
        # Reset counter
        self.output_count = 0
        start_time = rospy.Time.now()
        rate = rospy.Rate(30)  # Publish at 30 Hz
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            if (current_time - start_time).to_sec() > duration:
                break
            
            # Publish moving scenario
            t = (current_time - start_time).to_sec()
            
            # Varying velocity
            velocity = [2.0 * np.cos(t * 0.5), 0.5 * np.sin(t), 0.0]
            
            # Publish data
            self.publish_odometry([0, 0, 2], [0, 0, 0, 1])
            cloud = self.create_point_cloud(obstacle_points)
            self.cloud_pub.publish(cloud)
            self.publish_desired_velocity(velocity)
            
            rate.sleep()
        
        # Calculate statistics
        elapsed = (rospy.Time.now() - start_time).to_sec()
        avg_rate = self.output_count / elapsed
        
        rospy.loginfo(f"Outputs received: {self.output_count}")
        rospy.loginfo(f"Average rate: {avg_rate:.1f} Hz")
        
        return avg_rate

    def run_all_tests(self):
        """Run complete test suite"""
        rospy.loginfo(f"\n{'#'*50}")
        rospy.loginfo("VO FILTER NODE TEST SUITE (ROS)")
        rospy.loginfo(f"{'#'*50}")
        
        # Test 1: Direct collision
        self.run_test_case(
            "Direct Collision",
            "Obstacle directly ahead, should slow down",
            desired_velocity=[2.0, 0.0, 0.0],
            obstacle_points=[[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            expected_check=lambda v_in, v_out: v_out[0] < v_in[0] * 0.8
        )
        
        rospy.sleep(0.5)
        
        # Test 2: Clear path
        self.run_test_case(
            "Clear Path",
            "No obstacles nearby, velocity unchanged",
            desired_velocity=[2.0, 0.0, 0.0],
            obstacle_points=[[10.0, 10.0, 0.0]],  # Far away
            expected_check=lambda v_in, v_out: np.allclose(v_in, v_out, atol=0.2)
        )
        
        rospy.sleep(0.5)
        
        # Test 3: Obstacle to the right
        self.run_test_case(
            "Avoid Right",
            "Obstacle on right, should veer left",
            desired_velocity=[2.0, 0.0, 0.0],
            obstacle_points=[[2.0, 0.8, 0.0], [2.5, 0.8, 0.0]],
            expected_check=lambda v_in, v_out: v_out[1] < -0.1  # Negative Y is left
        )
        
        rospy.sleep(0.5)
        
        # Test 4: Zero velocity
        self.run_test_case(
            "Zero Velocity",
            "Not moving, should stay still",
            desired_velocity=[0.0, 0.0, 0.0],
            obstacle_points=[[1.0, 0.0, 0.0]],
            expected_check=lambda v_in, v_out: np.linalg.norm(v_out) < 0.1
        )
        
        rospy.sleep(0.5)
        
        # Test 5: Speed limit
        self.run_test_case(
            "Speed Limit",
            "High speed input should be capped",
            desired_velocity=[5.0, 0.0, 0.0],
            obstacle_points=[],  # No obstacles
            expected_check=lambda v_in, v_out: np.linalg.norm(v_out) <= 3.1  # max_speed = 3.0
        )
        
        # Performance test
        rospy.sleep(1.0)
        avg_rate = self.run_performance_test(duration=5.0)
        
        # Summary
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo("SUMMARY")
        rospy.loginfo(f"{'='*50}")
        
        passed = sum(1 for r in self.test_results if r.get('passed', False))
        total = len(self.test_results)
        
        rospy.loginfo(f"Correctness: {passed}/{total} tests passed")
        for result in self.test_results:
            status = "✓" if result.get('passed', False) else "✗"
            rospy.loginfo(f"  {status} {result['name']}")
        
        rospy.loginfo(f"\nPerformance: {avg_rate:.1f} Hz average")
        
        if avg_rate > 30:
            rospy.loginfo("✓ Performance is GOOD for real-time control")
        else:
            rospy.logwarn("✗ Performance is TOO SLOW for real-time control")
        
        if passed == total and avg_rate > 30:
            rospy.loginfo("\n✓✓✓ VO FILTER NODE IS WORKING CORRECTLY! ✓✓✓")
        else:
            rospy.logwarn(f"\n✗ Issues found: {total-passed} failed tests or poor performance")

def main():
    try:
        tester = VONodeROSTester()
        tester.run_all_tests()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()