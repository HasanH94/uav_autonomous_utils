#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
import tf

class TrajectoryTranslator:
    def __init__(self):
        rospy.init_node('pva_to_trajectory_translator', anonymous=True)
        
        self.trajectory_pub = rospy.Publisher('/mavros/setpoint_trajectory/local', MultiDOFJointTrajectory, queue_size=1)
        
        rospy.Subscriber('/pva_setpoint', JointTrajectoryPoint, self.pva_callback)
        
        rospy.loginfo("PVA to MultiDOFJointTrajectory translator started.")

    def pva_callback(self, pva_msg):
        # Create the main MultiDOFJointTrajectory message
        traj = MultiDOFJointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = 'odom' # Or your relevant frame
        traj.joint_names.append('base_link') # Or your robot's base link

        # Create a single MultiDOFJointTrajectoryPoint
        point = MultiDOFJointTrajectoryPoint()

        # Create and populate the transforms (for position)
        transform = Transform()
        transform.translation.x = pva_msg.positions[0]
        transform.translation.y = pva_msg.positions[1]
        transform.translation.z = pva_msg.positions[2]
        
        # Assuming yaw is not provided by the planner, use a fixed orientation
        # The 4th position value in pva_msg is yaw, but often unused.
        # If it were used, we would convert it to a quaternion here.
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        transform.rotation.x = q[0]
        transform.rotation.y = q[1]
        transform.rotation.z = q[2]
        transform.rotation.w = q[3]

        # Create and populate the velocities
        velocities = Twist()
        velocities.linear.x = pva_msg.velocities[0]
        velocities.linear.y = pva_msg.velocities[1]
        velocities.linear.z = pva_msg.velocities[2]

        # Create and populate the accelerations
        accelerations = Twist()
        accelerations.linear.x = pva_msg.accelerations[0]
        accelerations.linear.y = pva_msg.accelerations[1]
        accelerations.linear.z = pva_msg.accelerations[2]

        # Assign the populated messages to the trajectory point
        point.transforms.append(transform)
        point.velocities.append(velocities)
        point.accelerations.append(accelerations)
        
        # Add the point to the trajectory
        traj.points.append(point)
        
        # Publish the final trajectory message
        self.trajectory_pub.publish(traj)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        translator = TrajectoryTranslator()
        translator.run()
    except rospy.ROSInterruptException:
        pass
