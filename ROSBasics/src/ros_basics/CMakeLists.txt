cmake_minimum_required(VERSION 3.0.2)
project(ros_basics)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  dynamic_reconfigure
)


## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/Rate.cfg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_basics
  CATKIN_DEPENDS roscpp rospy std_msgs dynamic_reconfigure
  DEPENDS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(cpp_publisher src/cpp_publisher.cpp)
target_link_libraries(cpp_publisher ${catkin_LIBRARIES})

add_executable(cpp_subscriber src/cpp_subscriber.cpp)
target_link_libraries(cpp_subscriber ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS scripts/py_publisher.py scripts/py_subscriber.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)