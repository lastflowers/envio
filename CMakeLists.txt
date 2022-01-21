cmake_minimum_required(VERSION 2.8.3)
project(ensemble_vio)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-5")
set(CMAKE_C_COMPILER "/usr/bin/gcc-5")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  genmsg
  nav_msgs
  tf
  message_filters
  cv_bridge
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(envio_node
    src/envio_node.cpp
    src/core/vision_meas.cpp
    src/core/sl_iekf.cpp
    src/utils/Attitude.cpp
    src/utils/utils.cpp)

target_link_libraries(envio_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES})
