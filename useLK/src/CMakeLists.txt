# set property
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

# OenpCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

# APP
add_executable(useLK useLK.cpp)
target_link_libraries(useLK ${OpenCV_LIBS})
