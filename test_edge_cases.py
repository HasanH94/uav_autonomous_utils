#!/usr/bin/env python3
"""
Edge case testing for state machine
"""

import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped

def test_rapid_mode_changes():
    """Test rapid ArUco detection/loss"""
    rospy.init_node('edge_case_tester', anonymous=True)
    
    aruco_pub = rospy.Publisher('/aruco_detection_status', Bool, queue_size=10)
    pose_pub = rospy.Publisher('/mavros/local_position/pose', PoseStamped, queue_size=10)
    
    rospy.sleep(1.0)
    
    rospy.loginfo("Testing rapid ArUco detection changes...")
    
    # Set drone near GPS goal
    pose = PoseStamped()
    pose.header.frame_id = "odom"
    pose.pose.position.x = 9.5
    pose.pose.position.y = -5.0
    pose.pose.position.z = 3.0
    pose.pose.orientation.w = 1.0
    pose_pub.publish(pose)
    
    # Rapidly toggle ArUco detection
    for cycle in range(5):
        rospy.loginfo(f"Cycle {cycle+1}: Detecting ArUco...")
        # Send detections
        for i in range(35):  # More than window size
            status = Bool()
            status.data = True
            aruco_pub.publish(status)
            rospy.sleep(0.02)  # 50Hz
        
        rospy.loginfo(f"Cycle {cycle+1}: Losing ArUco...")
        # Send losses
        for i in range(35):
            status = Bool()
            status.data = False
            aruco_pub.publish(status)
            rospy.sleep(0.02)
        
        rospy.sleep(0.5)  # Brief pause between cycles
    
    rospy.loginfo("Rapid change test complete")

def test_null_visual_target():
    """Test behavior when visual target is never set"""
    rospy.init_node('null_target_tester', anonymous=True)
    
    # Just send ArUco detection status without pose
    aruco_pub = rospy.Publisher('/aruco_detection_status', Bool, queue_size=10)
    
    rospy.sleep(1.0)
    
    rospy.loginfo("Testing ArUco detection without pose...")
    
    for i in range(50):
        status = Bool()
        status.data = True
        aruco_pub.publish(status)
        rospy.sleep(0.05)
    
    rospy.loginfo("Null target test complete")

def test_simultaneous_goals():
    """Test publishing both GPS and visual goals simultaneously"""
    rospy.init_node('simultaneous_goal_tester', anonymous=True)
    
    mission_pub = rospy.Publisher('/move_base_mission', PoseStamped, queue_size=10)
    aruco_pub = rospy.Publisher('/aruco_offset_pose', PoseStamped, queue_size=10)
    
    rospy.sleep(1.0)
    
    rospy.loginfo("Publishing simultaneous conflicting goals...")
    
    for i in range(10):
        # GPS goal
        gps_goal = PoseStamped()
        gps_goal.header.stamp = rospy.Time.now()
        gps_goal.header.frame_id = "odom"
        gps_goal.pose.position.x = 10.0
        gps_goal.pose.position.y = -5.0
        gps_goal.pose.position.z = 3.0
        gps_goal.pose.orientation.w = 1.0
        mission_pub.publish(gps_goal)
        
        # Visual goal at different location
        visual_goal = PoseStamped()
        visual_goal.header.stamp = rospy.Time.now()
        visual_goal.header.frame_id = "odom"
        visual_goal.pose.position.x = 8.0  # Different location
        visual_goal.pose.position.y = -3.0
        visual_goal.pose.position.z = 2.5
        visual_goal.pose.orientation.w = 1.0
        aruco_pub.publish(visual_goal)
        
        rospy.sleep(0.1)
    
    rospy.loginfo("Simultaneous goal test complete")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        if test_name == "rapid":
            test_rapid_mode_changes()
        elif test_name == "null":
            test_null_visual_target()
        elif test_name == "simultaneous":
            test_simultaneous_goals()
        else:
            rospy.logerr(f"Unknown test: {test_name}")
            rospy.loginfo("Available tests: rapid, null, simultaneous")
    else:
        rospy.loginfo("Running all edge case tests...")
        
        # Run each test with a fresh node
        test_rapid_mode_changes()
        rospy.signal_shutdown("Test complete")
        rospy.sleep(2.0)
        
        rospy.init_node('edge_case_tester_2', anonymous=True)
        test_null_visual_target()
        rospy.signal_shutdown("Test complete")
        rospy.sleep(2.0)
        
        rospy.init_node('edge_case_tester_3', anonymous=True)
        test_simultaneous_goals()
        
        rospy.loginfo("All edge case tests complete!")