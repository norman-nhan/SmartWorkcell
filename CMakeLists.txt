cmake_minimum_required(VERSION 3.0.2)
project(smartworkcell)

find_package(catkin REQUIRED COMPONENTS 
  rospy 
  std_msgs 
  geometry_msgs
  sensor_msgs 
  cv_bridge
  message_generation
  actionlib
  actionlib_msgs
)
catkin_python_setup()  # This enables setup.py integration


################################################
## Declare ROS messages, services and actions ##
################################################

add_service_files(
  FILES
  DetectObjectPose.srv
)
add_action_files(
  DIRECTORY action
  FILES MoveJoint2ObjPose.action
)
## Generate added messages and services, actions
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
)


###################################
## catkin specific configuration ##
###################################


catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs message_runtime
)
catkin_install_python(PROGRAMS
  src/ObjectDetectNode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Install ##
#############
install(PROGRAMS
  src/ObjectDetectNode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
