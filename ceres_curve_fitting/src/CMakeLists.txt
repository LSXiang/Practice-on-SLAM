# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++11")

# 添加cmake模块以使用ceres库
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# message(STATUS "${PROJECT_SOURCE_DIR}") 

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(ceres_curve_fitting ceres_curve_fitting.cpp)
target_link_libraries(ceres_curve_fitting ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
