find_package(catkin REQUIRED COMPONENTS
  rospy std_msgs geometry_msgs nav_msgs sensor_msgs message_generation trajectory_msgs
)

add_message_files(FILES
  SupportPoint.msg
  SupportArray.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs)