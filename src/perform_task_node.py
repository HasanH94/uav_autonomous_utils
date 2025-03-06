#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import time

def perform_task():
    # Initialize the ROS node
    rospy.init_node('perform_task_node', anonymous=True)
    
    # Create a publisher to notify the State Manager when the task is done
    task_done_pub = rospy.Publisher('/task_done', Bool, queue_size=10)
    
    rospy.loginfo("Perform Task Node Started. Performing task...")
    
    # Define the task duration (in seconds)
    task_duration = 10  # 10 seconds
    
    # Determine if ROS is using simulated time
    use_sim_time = rospy.get_param('/use_sim_time', False)
    
    if use_sim_time:
        rospy.loginfo("Simulated time is enabled. Using rospy.get_time().")
        
        # Wait until ROS time is non-zero (i.e., /clock is being published)
        while rospy.get_time() == 0.0 and not rospy.is_shutdown():
            rospy.loginfo("Waiting for ROS time to be initialized...")
            rospy.sleep(0.1)
        
        start_time = rospy.get_time()
    else:
        rospy.loginfo("Simulated time is disabled. Using system time.")
        start_time = time.time()
    
    # Define the rate at which to publish False messages
    rate = rospy.Rate(10)  # 10 Hz
    
    # Continuously publish False until the task is completed
    while not rospy.is_shutdown():
        if use_sim_time:
            current_time = rospy.get_time()
        else:
            current_time = time.time()
        
        elapsed_time = current_time - start_time
        
        if elapsed_time < task_duration:
            # Publish False to indicate task is ongoing
            task_done_msg = Bool(data=False)
            task_done_pub.publish(task_done_msg)
            rospy.logdebug(f"Task ongoing: {elapsed_time:.2f}/{task_duration} seconds elapsed.")
        else:
            # Publish True to indicate task completion
            task_done_msg = Bool(data=True)
            task_done_pub.publish(task_done_msg)
            rospy.loginfo("Perform Task Completed. Notifying State Manager.")
            
            # Allow some time for the message to be sent
            rospy.sleep(1)
            
            # Shutdown the node
            rospy.signal_shutdown("Task Completed")
            break
        
        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        perform_task()
    except rospy.ROSInterruptException:
        pass
