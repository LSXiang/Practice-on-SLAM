find_package(Sophus REQUIRED)
include_directories("/usr/include/eigen3")
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus ${Sophus_LIBRARIES})
