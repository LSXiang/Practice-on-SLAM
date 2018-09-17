# set the minimum required version of cmake for the project
cmake_minimum_required(VERSION 2.8)

# project name 
project(PoseGraph)

# set property
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++0x -march=native -O3")

# append elements to the CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIR})

# gtsam
# find_package(GTSAMCMakeTools)
# find_package(GTSAM REQUIRED)
# include_directories(${GTSAM_INCLUDE_DIR})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(src)
