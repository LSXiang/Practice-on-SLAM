add_executable(dense_mapping_monocular dense_mapping_monocular.cpp)
target_link_libraries(dense_mapping_monocular ${OpenCV_LIBS} ${Sophus_LIBRARIES})

add_executable(dense_mapping_monocular2 dense_mapping_monocular_2.cpp)
target_link_libraries(dense_mapping_monocular2 ${OpenCV_LIBS} ${Sophus_LIBRARIES})
