# set property
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug")

# add cmake modules 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV REQUIRED)

# G2O
find_package(G2O REQUIRED)

# include directories
include_directories(${OpenCV_DIRS} ${G2O_INCLUDE_DIR} "/usr/include/eigen3")

set(G2O_LIBS g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension)

# set ignore "deprecated-declarations" warning
add_compile_options("-Wno-deprecated-declarations")

# APP 
add_executable(direct_sparse direct_sparse.cpp)
target_link_libraries(direct_sparse ${OpenCV_LIBS} ${G2O_LIBS})
