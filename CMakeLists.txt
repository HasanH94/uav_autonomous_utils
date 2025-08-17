cmake_minimum_required(VERSION 3.8)
project(uav_autonomous_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  nav_msgs
  trajectory_msgs
  message_generation
  # Add other ROS 1 dependencies here
)

find_package(PCL REQUIRED)

# Generate messages
add_message_files(FILES
  SupportPoint.msg
  SupportArray.msg
)

# Generate services
add_service_files(FILES
  SetNavigationMode.srv
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
  uav_autonomous_utils
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

message(STATUS "PCL include directories: ${PCL_INCLUDE_DIRS} !!!!!!!!!!!!!!!!!!!!!!!!")



# find dependencies
##find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

##if(BUILD_TESTING)
  ##find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ##ament_lint_auto_find_test_dependencies()
##endif()

##ament_package()
