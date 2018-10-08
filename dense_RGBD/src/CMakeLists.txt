
add_executable(point_cloud_mapping point_cloud_mapping.cpp)
target_link_libraries(point_cloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(octomap_mapping octomap_mapping.cpp)
target_link_libraries(octomap_mapping ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES})
