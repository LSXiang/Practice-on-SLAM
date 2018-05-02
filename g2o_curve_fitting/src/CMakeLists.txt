set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# g2o
# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
# 
# message(STATUS "${G2O_INCLUDE_DIR} ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY}")

add_executable(g2o_curve_fitting g2o_curve_fitting.cpp)
target_link_libraries(g2o_curve_fitting ${OpenCV_LIBRARIES} ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY})
