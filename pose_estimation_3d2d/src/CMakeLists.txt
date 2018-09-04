# set property
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

# add cmake modules 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV REQUIRED)
# g2o
find_package(G2O REQUIRED)
# csparse
find_package(CSparse REQUIRED)
# Ceres
find_package(Ceres REQUIRED)

# include
include_directories(${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} "/usr/include/eigen3/" ${CERES_INCLUDE_DIRS})

# set ignore "deprecated-declarations" warning
add_compile_options("-Wno-deprecated-declarations")

# App 
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries( pose_estimation_3d2d 
    ${OpenCV_LIBRARIES} 
    ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_TYPES_SBA} ${G2O_SOLVER_CSPARSE_EXTENSION}
    ${CSPARSE_LIBRARY}
)

add_executable(pose_estimation_3d2d_ceres pose_estimation_3d2d_ceres.cpp)
target_link_libraries( pose_estimation_3d2d_ceres
    ${OpenCV_LIBRARIES}
    ${CERES_LIBS}
)


