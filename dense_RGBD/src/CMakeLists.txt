
add_executable(point_cloud_mapping point_cloud_mapping.cpp)
target_link_libraries(point_cloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})
