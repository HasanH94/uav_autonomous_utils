#!/usr/bin/env python3
import rospy
import roslaunch
import subprocess
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Header
import math
import sys
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from geometry_msgs.msg import Twist  # Ensure this is imported

# Initialize global variables
goal_position = None
current_position = None
landed = False
current_process = None

# Service proxies
set_mode_client = None
arming_client = None

# Goal threshold
GOAL_THRESHOLD = 0.2  # meters

# Initialize the ROS node first
def initialize_node():
    rospy.init_node("test_node", anonymous=True)
    rospy.loginfo("[Initialization] ROS node 'test_node' initialized.")

# Wait for Mavros services after initializing the node
def initialize_services():
    global set_mode_client, arming_client
    try:
        rospy.loginfo("[Initialization] Waiting for Mavros services '/mavros/set_mode' and '/mavros/cmd/arming'...")
        rospy.wait_for_service('/mavros/set_mode', timeout=30)
        rospy.wait_for_service('/mavros/cmd/arming', timeout=30)
        rospy.loginfo("[Initialization] Mavros services are available.")
    except rospy.ROSException:
        rospy.logerr("[Initialization] Timeout while waiting for Mavros services.")
        sys.exit(1)

    set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

# Calculate Euclidean distance between two positions
def calculate_distance(pos1, pos2):
    return math.sqrt(
        (pos1.x - pos2.x) ** 2 +
        (pos1.y - pos2.y) ** 2 +
        (pos1.z - pos2.z) ** 2
    )

# Stop any currently running process or launch file
def stop_current_process(current_process):
    """Stop any currently running process or launch file."""
    if current_process:
        rospy.loginfo(f"[StateManager] Stopping current process: {current_process}")
        try:
            if isinstance(current_process, roslaunch.parent.ROSLaunchParent):
                current_process.shutdown()
                rospy.loginfo("[StateManager] roslaunch parent shutdown complete.")
            else:
                current_process.terminate()
                current_process.wait()
                rospy.loginfo("[StateManager] Terminated subprocess.")
        except Exception as e:
            rospy.logerr(f"[StateManager] Error stopping process: {e}")
        finally:
            current_process = None
    else:
        rospy.loginfo("[StateManager] No current process to stop.")

# Publish a PoseStamped message to /move_base_simple/goal
def publish_goal(x, y, z):
    """Publish a PoseStamped message to /move_base_simple/goal."""
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    
    # Allow some time for the publisher to register with subscribers
    rospy.sleep(1)
    
    # Wait until at least one subscriber is connected
    rate = rospy.Rate(10)  # 10 Hz
    while goal_pub.get_num_connections() < 1 and not rospy.is_shutdown():
        rospy.loginfo("[StateManager] Waiting for subscribers to connect to /move_base_simple/goal...")
        rate.sleep()
    
    # Create and populate the PoseStamped message
    goal = PoseStamped()
    goal.header = Header()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"  # Ensure this matches your navigation frame
    
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    
    # Orientation (optional): Facing forward (identity quaternion)
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0
    
    # Publish the goal
    goal_pub.publish(goal)
    rospy.loginfo(f"[StateManager] Published goal to /move_base_simple/goal: x={x}, y={y}, z={z}")

# Launch a .launch file (e.g., local_node.launch) with arguments
def launch_file(package, launch_file, x, y, z):
    """Launch a .launch file (e.g., local_node.launch) with arguments."""
    global current_process
    rospy.loginfo(f"[StateManager] Attempting to launch file: {package}/{launch_file}")
    stop_current_process(current_process)
    try:
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]
        
        # Prepare launch arguments as a list of tuples
        launch_args = [f"goal_x:={x}", f"goal_y:={y}", f"goal_z:={z}"]
        roslaunch_files = [(launch_path, launch_args)]
        
        current_process = roslaunch.parent.ROSLaunchParent(
            uuid, roslaunch_files,
            sigint_timeout=1.0, sigterm_timeout=1.0
        )
        current_process.start()
        rospy.loginfo(f"[StateManager] Launched: {package}/{launch_file} with args {launch_args}")
        
    except Exception as e:
        rospy.logerr(f"[StateManager] Failed to launch {package}/{launch_file}: {e}")
    
    return current_process

# Set the flight mode to OFFBOARD
def set_offboard_mode():
    """Sets the flight mode to OFFBOARD."""
    mode = 'OFFBOARD'
    set_mode = SetModeRequest()
    set_mode.custom_mode = mode
    set_mode.base_mode = 0  # must be 0 when specifying custom_mode

    try:
        rospy.loginfo(f"[DroneStateMachine] Attempting to set mode to {mode}")
        response = set_mode_client(set_mode)
        if response.mode_sent:
            rospy.loginfo("[DroneStateMachine] OFFBOARD mode set successfully.")
            return True
        else:
            rospy.logwarn("[DroneStateMachine] Failed to set OFFBOARD mode.")
            return False
    except rospy.ServiceException as e:
        rospy.logerr(f"[DroneStateMachine] Service call failed while setting mode: {e}")
        return False

# Set the flight mode to AUTO.LAND to initiate landing
def set_land_mode():
    """Sets the flight mode to AUTO.LAND to initiate landing."""
    mode = 'AUTO.LAND'
    set_mode = SetModeRequest()
    set_mode.custom_mode = mode
    set_mode.base_mode = 0

    try:
        rospy.loginfo(f"[DroneStateMachine] Attempting to set mode to {mode}")
        response = set_mode_client(set_mode)
        if response.mode_sent:
            rospy.loginfo(f"[DroneStateMachine] {mode} mode set successfully.")
        else:
            rospy.logwarn(f"[DroneStateMachine] Failed to set {mode} mode.")
    except rospy.ServiceException as e:
        rospy.logerr(f"[DroneStateMachine] Service call failed while setting mode to {mode}: {e}")

# Arm the drone
def arm_drone():
    """Arms the drone."""
    try:
        rospy.loginfo("[DroneStateMachine] Attempting to arm the drone.")
        response = arming_client(True)  # True to arm
        if response.success:
            rospy.loginfo("[DroneStateMachine] Drone armed successfully.")
            return True
        else:
            rospy.logwarn("[DroneStateMachine] Failed to arm the drone.")
            return False
    except rospy.ServiceException as e:
        rospy.logerr(f"[DroneStateMachine] Service call failed while arming: {e}")
        return False

# Publish initial setpoints to enable OFFBOARD mode
def publish_initial_setpoints():
    """Publish initial setpoints to enable OFFBOARD mode."""
    cmd_vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(20)  # 20 Hz
    twist = Twist()
    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0
    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0
    
    rospy.loginfo("[StateManager] Publishing initial setpoints...")
    for _ in range(100):
        if rospy.is_shutdown():
            break
        cmd_vel_pub.publish(twist)
        rate.sleep()
    rospy.loginfo("[StateManager] Initial setpoints published.")

# Handle shutdown gracefully
def shutdown_hook(current_process):
    """Handles cleanup during shutdown."""
    rospy.loginfo("[StateManager] Shutting down. Stopping current processes and landing drone...")
    stop_current_process(current_process)
    set_land_mode()

# Subscriber callback for current position
def position_callback(msg):
    global current_position, goal_position, landed
    current_position = msg.pose.position

    if goal_position is None:
        #rospy.logwarn("[DroneStateMachine] Goal position is not set.")
        return

    distance = calculate_distance(current_position, goal_position)
    #rospy.loginfo(f"[DroneStateMachine] Current distance to goal: {distance:.3f} meters.")

    if distance < GOAL_THRESHOLD and not landed:
        rospy.loginfo("[DroneStateMachine] Goal reached within threshold. Initiating landing.")
        set_land_mode()
        landed = True

def main():
    global goal_position  # Ensure goal_position is global
    initialize_node()
    initialize_services()
    
    # Subscribe to the drone's current position
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, position_callback)
    
    goal_points = [
        (1, 2, 3),
        (-4, -5, 3),
        (0, 0, 2)
    ]
    counter = 0
    current_process = None
    
    # Register shutdown hook
    rospy.on_shutdown(lambda: shutdown_hook(current_process))
    
    # Launch the first goal
    current_process = launch_file("local_planner", "local_node.launch",
                                  goal_points[counter][0],
                                  goal_points[counter][1],
                                  goal_points[counter][2])
    
    # Publish initial setpoints
    publish_initial_setpoints()
    
    # Set OFFBOARD mode
    if not set_offboard_mode():
        rospy.logwarn("[Main] Attempting to set OFFBOARD mode failed. Retrying after a short delay...")
        rospy.sleep(5)
        set_offboard_mode()
    
    # Arm the drone
    if not arm_drone():
        rospy.logwarn("[Main] Attempting to arm the drone failed. Retrying after a short delay...")
        rospy.sleep(5)
        arm_drone()
    
    # Publish the goal
    x, y, z = goal_points[counter]
    goal_position = PoseStamped().pose.position
    goal_position.x = x
    goal_position.y = y
    goal_position.z = z
    publish_goal(x, y, z)
    
    # Keep the script running to maintain the node and launch processes
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[Main] KeyboardInterrupt received. Shutting down.")
    
    # Upon shutdown, stop the current process and land
    stop_current_process(current_process)
    set_land_mode()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
