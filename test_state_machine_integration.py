#!/usr/bin/env python3
"""
Test script to verify the enhanced state machine integration
Checks for potential issues, race conditions, and edge cases
"""

import rospy
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String, Float32
from threading import Lock

class StateMachineIntegrationTest:
    def __init__(self):
        rospy.init_node('state_machine_test', anonymous=True)
        
        # Test results
        self.test_results = {
            'mode_published': False,
            'distance_published': False,
            'gps_goal_published': False,
            'visual_goal_published': False,
            'mode_changes': [],
            'state_changes': [],
            'last_mode': None,
            'last_state': None,
            'goal_publish_times': [],
            'distance_values': []
        }
        
        self.lock = Lock()
        
        # Subscribe to all outputs from state machine
        rospy.Subscriber('/navigation/current_mode', String, self.mode_callback)
        rospy.Subscriber('/navigation/target_distance', Float32, self.distance_callback)
        rospy.Subscriber('/move_base_gps', PoseStamped, self.gps_goal_callback)
        rospy.Subscriber('/move_base_visual', PoseStamped, self.visual_goal_callback)
        rospy.Subscriber('/state_machine/current_state', String, self.state_callback)
        
        # Publishers for testing inputs
        self.aruco_pub = rospy.Publisher('/aruco_detection_status', Bool, queue_size=10)
        self.aruco_pose_pub = rospy.Publisher('/aruco_offset_pose', PoseStamped, queue_size=10)
        self.pose_pub = rospy.Publisher('/mavros/local_position/pose', PoseStamped, queue_size=10)
        
        rospy.sleep(1.0)  # Wait for connections
        
    def mode_callback(self, msg):
        with self.lock:
            self.test_results['mode_published'] = True
            if self.test_results['last_mode'] != msg.data:
                self.test_results['mode_changes'].append({
                    'time': rospy.Time.now().to_sec(),
                    'mode': msg.data
                })
                self.test_results['last_mode'] = msg.data
                rospy.loginfo(f"Mode changed to: {msg.data}")
    
    def state_callback(self, msg):
        with self.lock:
            if self.test_results['last_state'] != msg.data:
                self.test_results['state_changes'].append({
                    'time': rospy.Time.now().to_sec(),
                    'state': msg.data
                })
                self.test_results['last_state'] = msg.data
                rospy.loginfo(f"State changed to: {msg.data}")
    
    def distance_callback(self, msg):
        with self.lock:
            self.test_results['distance_published'] = True
            self.test_results['distance_values'].append(msg.data)
    
    def gps_goal_callback(self, msg):
        with self.lock:
            self.test_results['gps_goal_published'] = True
            self.test_results['goal_publish_times'].append(rospy.Time.now().to_sec())
            rospy.loginfo(f"GPS goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")
    
    def visual_goal_callback(self, msg):
        with self.lock:
            self.test_results['visual_goal_published'] = True
            rospy.loginfo(f"Visual goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})")
    
    def simulate_drone_pose(self, x, y, z):
        """Simulate drone at a specific position"""
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "odom"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
    
    def simulate_aruco_detection(self, detected, x=10.0, y=-5.0, z=3.0):
        """Simulate ArUco marker detection"""
        # Publish detection status
        status = Bool()
        status.data = detected
        self.aruco_pub.publish(status)
        
        # If detected, publish pose
        if detected:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "odom"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            self.aruco_pose_pub.publish(pose)
    
    def run_tests(self):
        rospy.loginfo("="*50)
        rospy.loginfo("Starting State Machine Integration Tests")
        rospy.loginfo("="*50)
        
        # Test 1: Check initial publishing
        rospy.loginfo("\nTest 1: Checking initial topic publishing...")
        rospy.sleep(2.0)
        
        with self.lock:
            if self.test_results['mode_published']:
                rospy.loginfo("✓ Navigation mode is being published")
            else:
                rospy.logwarn("✗ Navigation mode NOT published")
            
            if self.test_results['distance_published']:
                rospy.loginfo("✓ Target distance is being published")
            else:
                rospy.logwarn("✗ Target distance NOT published (might need pose first)")
        
        # Test 2: Simulate drone movement and check distance updates
        rospy.loginfo("\nTest 2: Testing distance calculation...")
        self.simulate_drone_pose(0, 0, 2)
        rospy.sleep(0.5)
        
        initial_distance_count = len(self.test_results['distance_values'])
        self.simulate_drone_pose(1, 1, 2)
        rospy.sleep(0.5)
        
        with self.lock:
            if len(self.test_results['distance_values']) > initial_distance_count:
                rospy.loginfo(f"✓ Distance updates on pose change. Latest: {self.test_results['distance_values'][-1]:.2f}m")
            else:
                rospy.logwarn("✗ Distance not updating with pose changes")
        
        # Test 3: Check goal publishing frequency
        rospy.loginfo("\nTest 3: Checking goal publishing frequency...")
        with self.lock:
            self.test_results['goal_publish_times'] = []
        
        rospy.sleep(1.0)  # Collect 1 second of data
        
        with self.lock:
            if len(self.test_results['goal_publish_times']) >= 2:
                intervals = []
                for i in range(1, len(self.test_results['goal_publish_times'])):
                    intervals.append(self.test_results['goal_publish_times'][i] - self.test_results['goal_publish_times'][i-1])
                avg_interval = sum(intervals) / len(intervals) if intervals else 0
                rospy.loginfo(f"✓ Goals published {len(self.test_results['goal_publish_times'])} times in 1s (avg interval: {avg_interval:.3f}s)")
            else:
                rospy.logwarn(f"✗ Goals published only {len(self.test_results['goal_publish_times'])} times in 1s")
        
        # Test 4: Simulate ArUco detection
        rospy.loginfo("\nTest 4: Testing ArUco detection handling...")
        
        # Move drone closer to GPS goal first
        self.simulate_drone_pose(9, -5, 3)
        rospy.sleep(0.5)
        
        # Send consistent ArUco detections (need 70% of 30 frames)
        rospy.loginfo("Sending 30 ArUco detections to build confidence...")
        for i in range(30):
            self.simulate_aruco_detection(True, 10.0, -5.0, 3.0)
            rospy.sleep(0.05)  # ~20Hz
        
        rospy.sleep(1.0)
        
        with self.lock:
            # Check if visual goal was published
            if self.test_results['visual_goal_published']:
                rospy.loginfo("✓ Visual goal published after ArUco detection")
            else:
                rospy.logwarn("✗ Visual goal NOT published (might need to be closer or near GPS goal)")
            
            # Check mode changes
            if len(self.test_results['mode_changes']) > 0:
                rospy.loginfo(f"✓ Mode changes detected: {[m['mode'] for m in self.test_results['mode_changes']]}")
            else:
                rospy.logwarn("✗ No mode changes detected")
        
        # Test 5: Simulate ArUco loss
        rospy.loginfo("\nTest 5: Testing ArUco loss handling...")
        for i in range(35):  # Send enough false detections to lose confidence
            self.simulate_aruco_detection(False)
            rospy.sleep(0.05)
        
        rospy.sleep(1.0)
        
        with self.lock:
            if len(self.test_results['mode_changes']) > 1:
                last_mode = self.test_results['mode_changes'][-1]['mode']
                rospy.loginfo(f"✓ Mode changed after ArUco loss. Current mode: {last_mode}")
            else:
                rospy.logwarn("✗ No mode change after ArUco loss")
        
        # Print summary
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("Test Summary:")
        rospy.loginfo("="*50)
        
        with self.lock:
            rospy.loginfo(f"Total mode changes: {len(self.test_results['mode_changes'])}")
            rospy.loginfo(f"Total state changes: {len(self.test_results['state_changes'])}")
            rospy.loginfo(f"Distance values received: {len(self.test_results['distance_values'])}")
            
            if self.test_results['state_changes']:
                rospy.loginfo(f"States visited: {[s['state'] for s in self.test_results['state_changes']]}")
            if self.test_results['mode_changes']:
                rospy.loginfo(f"Modes visited: {[m['mode'] for m in self.test_results['mode_changes']]}")
        
        return self.test_results

if __name__ == '__main__':
    try:
        tester = StateMachineIntegrationTest()
        rospy.sleep(2.0)  # Give state machine time to initialize
        results = tester.run_tests()
        
        # Keep node alive to observe any delayed behaviors
        rospy.loginfo("\nMonitoring for 5 more seconds...")
        rospy.sleep(5.0)
        
        rospy.loginfo("\nTest complete!")
        
    except rospy.ROSInterruptException:
        pass