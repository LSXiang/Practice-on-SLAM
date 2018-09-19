
add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3 g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES})

add_executable(pose_graph_g2o_lie_algebra pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie_algebra g2o_core g2o_stuff ${CHOLMOD_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(pose_graph_gtsam pose_graph_gtsam.cpp)
target_link_libraries(pose_graph_gtsam ${CHOLMOD_LIBRARIES} gtsam)
