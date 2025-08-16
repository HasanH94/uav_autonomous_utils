#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest,SetModeRequest
from scipy.interpolate import CubicSpline
import heapq  # For implementing A* efficiently

class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position  # Position as (x, y, z)
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class DronePlanner:
    def __init__(self):
        rospy.init_node('planner_px4', anonymous=True)
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=1)
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        self.current_state = State()
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_raw", Float64MultiArray, self.depth_cb)

        # Initialize parameters for path planning and trajectory
        self.grid_map = self.create_dummy_grid_map(50, 50, 1)
        self.start_pos = (1, 5, 1)
        self.goal_pos = (49, 5, 1)
        self.t_sample = 0.05
        self.trajectory = []

    def state_cb(self, msg):
        self.current_state = msg

    def depth_cb(self, msg):
        # Example: Convert depth image to a grid map for dynamic obstacle detection
        depth_data = np.array(msg.data).reshape(msg.layout.dim[0].size, msg.layout.dim[1].size)
        self.update_grid_map_with_obstacles(depth_data)

    def create_dummy_grid_map(self, width, height, resolution):
        return np.zeros((int(height / resolution), int(width / resolution)))

    def update_grid_map_with_obstacles(self, depth_data):
        # Convert depth data into obstacles on the grid map
        rospy.loginfo("Updating grid map with obstacles")
        for i in range(depth_data.shape[0]):
            for j in range(depth_data.shape[1]):
                if depth_data[i, j] < 1.0:  # Arbitrary threshold for obstacle detection
                    self.grid_map[i, j] = 1

    def heuristic(self, node, goal):
        # Use Euclidean distance as the heuristic
        return np.linalg.norm(np.array(node) - np.array(goal))

    def run_kinodynamic_astar(self):
        start_node = Node(self.start_pos, h=self.heuristic(self.start_pos, self.goal_pos))
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_list = set()

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)

            if current_node.position == self.goal_pos:
                rospy.loginfo("Goal reached!")
                return self.reconstruct_path(current_node)

            neighbors = self.get_neighbors(current_node.position)
            for neighbor in neighbors:
                if neighbor in closed_list:
                    continue

                g = current_node.g + np.linalg.norm(np.array(neighbor) - np.array(current_node.position))
                h = self.heuristic(neighbor, self.goal_pos)
                neighbor_node = Node(neighbor, g=g, h=h, parent=current_node)

                heapq.heappush(open_list, neighbor_node)

        raise Exception("Path not found")

    def get_neighbors(self, position):
        x, y, z = position
        neighbors = [
            (x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        valid_neighbors = [n for n in neighbors if self.is_valid(n)]
        return valid_neighbors

    def is_valid(self, position):
        x, y, z = position
        return 0 <= x < self.grid_map.shape[1] and 0 <= y < self.grid_map.shape[0] and self.grid_map[int(y), int(x)] == 0

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        path.reverse()
        return path

    def parameterize_trajectory(self, path):
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] for p in path]
        t = np.linspace(0, len(path) * self.t_sample, len(path))

        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)
        spline_z = CubicSpline(t, z)

        trajectory = []
        t_new = np.linspace(0, t[-1], int(t[-1] / self.t_sample))
        for t_i in t_new:
            trajectory.append((spline_x(t_i), spline_y(t_i), spline_z(t_i)))
        return trajectory

    def execute_trajectory(self):
        pose = PoseStamped()
        for point in self.trajectory:
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
            self.local_pos_pub.publish(pose)
            rospy.sleep(self.t_sample)

    def arm_and_set_offboard_mode(self):
        rospy.loginfo("Arming the drone and setting to OFFBOARD mode...")
        offb_set_mode = SetModeRequest()
        offb_set_mode.base_mode = 0  # Optional, set to 0 if not needed
        offb_set_mode.custom_mode = "OFFBOARD"

        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        last_req = rospy.Time.now()

        while not rospy.is_shutdown() and not self.current_state.armed:
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req > rospy.Duration(5.0)):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD mode enabled")
                last_req = rospy.Time.now()

            if not self.current_state.armed and (rospy.Time.now() - last_req > rospy.Duration(5.0)):
                if self.arming_client.call(arm_cmd).success:
                    rospy.loginfo("Vehicle armed")
                last_req = rospy.Time.now()


    def start(self):
        rospy.loginfo("Starting drone planner...")
        path = self.run_kinodynamic_astar()
        self.trajectory = self.parameterize_trajectory(path)
        self.arm_and_set_offboard_mode()
        self.execute_trajectory()

if __name__ == "__main__":
    planner = DronePlanner()
    planner.start()
    rospy.spin()
