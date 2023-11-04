cmake_minimum_required(VERSION 3.2)
project(task1)

find_package(OpenCV  REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "   OpenCV Version: ${OpenCV_VERSION}")
include_directories("/usr/include/eigen3")

set(CMAKE_CXX_STANDARD 14)


add_executable(task1 main.cpp )
target_link_libraries(task1 ${OpenCV_LIBS} )


