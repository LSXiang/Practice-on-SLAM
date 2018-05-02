project(useGeometry)
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )
# 添加Eigen
add_executable(useGeometry useGeometry.cpp)
