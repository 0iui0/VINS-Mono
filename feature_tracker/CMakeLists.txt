cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -mmacosx-version-min=11.2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(catkin_INCLUDE_DIRS "/opt/ros/kinetic/include/")
include_directories("/usr/local/include/")
include_directories("/opt/homebrew/include/")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#find_package(catkin REQUIRED COMPONENTS
#        roscpp
#        std_msgs
#        sensor_msgs
#        cv_bridge
#        camera_model
#        )
#
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
#
#catkin_package()
#
#include_directories(
#        ${catkin_INCLUDE_DIRS}
#)
#
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
        src/feature_tracker_node.cpp
        src/parameters.cpp
        src/feature_tracker.cpp
        )

target_link_libraries(feature_tracker /Users/iouoi/CLionProjects/VINS-Mono/libros ${OpenCV_LIBS})
