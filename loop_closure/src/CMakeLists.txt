
add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBS})
