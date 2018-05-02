# set property
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#

# APP
add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBRARIES})
