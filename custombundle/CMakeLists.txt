##
# Copyright (c) 2018, The Akatsuki(Jacob.lsx). All rights reserved.
#

# set the minimum required version of cmake for the project
cmake_minimum_required(VERSION 2.8)

# project name 
project(customBundle)

# set property
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-march=native -O3")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)

# include common header file path
include_directories(${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/tools
                    ${PROJECT_SOURCE_DIR}/common/flags)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

# add subdirectories
add_subdirectory(custombundle_g2o)
add_subdirectory(custombundle_ceres)
