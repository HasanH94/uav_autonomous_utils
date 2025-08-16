import rospy
import roslaunch
import subprocess

class StateManager:
    def __init__(self):
        self.current_process = None
        rospy.init_node('state_manager_node', anonymous=True)

    def launch_file(self, package, launch_file):
        if self.current_process:
            self.stop_current_process()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_path = roslaunch.rlutil.resolve_launch_arguments([package, launch_file])[0]
        self.current_process = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        self.current_process.start()
        rospy.loginfo(f"Started launch file: {package}/{launch_file}")

    def run_node(self, package, node_type, node_name):
        if self.current_process:
            self.stop_current_process()
        command = f"rosrun {package} {node_type}"
        self.current_process = subprocess.Popen(command, shell=True)
        rospy.loginfo(f"Started node: {package}/{node_type}")

    def stop_current_process(self):
        if isinstance(self.current_process, roslaunch.parent.ROSLaunchParent):
            self.current_process.shutdown()
        elif isinstance(self.current_process, subprocess.Popen):
            self.current_process.terminate()
            self.current_process.wait()
        self.current_process = None
        rospy.loginfo("Stopped current process")

    def transition_to_state(self, state_name):
        if state_name == 'gps_navigation':
            self.launch_file('my_package', 'gps_navigation.launch')
        elif state_name == 'visual_servoing':
            self.run_node('my_package', 'visual_servoing_node', 'visual_servoing')
        elif state_name == 'performing_task':
            self.run_node('my_package', 'perform_task_node', 'perform_task')
        elif state_name == 'search_for_object':
            self.run_node('my_package', 'search_object_node', 'search_for_object')
        else:
            rospy.logwarn(f"Unknown state: {state_name}")

if __name__ == "__main__":
    manager = StateManager()
    # Example of switching states
    manager.transition_to_state('gps_navigation')
    rospy.sleep(5)  # Simulate some time in the state
    manager.transition_to_state('visual_servoing')
    rospy.sleep(5)
    manager.transition_to_state('performing_task')
