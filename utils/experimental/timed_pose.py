#!/usr/bin/env python3
import rospy
import roslaunch
import time

def times_pose_estimation():
    rospy.init_node('timed_node_controller', anonymous=True)
    
    launch_file = ('othmanPack', 'pose.launch') 
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    
    while not rospy.is_shutdown():
        # Start the launch
        parent = roslaunch.parent.ROSLaunchParent(
            uuid, 
            [roslaunch.rlutil.resolve_launch_arguments(list(launch_file))[0]]
        )
        rospy.loginfo("Starting estimation_of_pose...")
        parent.start()
        
        rospy.loginfo("Running for T seconds...")
        t = 60
        time.sleep(t)
        
        # Stop the launch
        rospy.loginfo("Shutting down estimation_of_pose...")
        parent.shutdown()
        
        rospy.loginfo("Node is OFF. Waiting T seconds...")
        time.sleep(1/t)



if __name__ == '__main__':
    pass
    #times_pose_estimation()
