cmake_minimum_required(VERSION 2.8)
project(loop_closure)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

# OpenCV
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# DBoW3
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_LIB_DIR})

add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")

add_subdirectory(src)
