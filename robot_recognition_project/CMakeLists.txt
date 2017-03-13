set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
cmake_minimum_required(VERSION 3.6)

project(robot_recognition_project)

find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (recognise main.cpp)
target_link_libraries (recognise ${PCL_LIBRARIES})