##
# Copyright (c) 2018, The Akatsuki(Jacob.lsx). All rights reserved.
#
# custombundle_g2o/CMakeLists.txt
#

find_package(G2O REQUIRED)
find_package(Cholmod REQUIRED)

set(G2O_LIBRARIES  g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ceres)
include_directories(${G2O_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} custombundle_g2o_src)


add_executable(customBundle_G2O ${custombundle_g2o_src})

target_link_libraries(customBundle_G2O ${G2O_LIBRARIES} ${CHOLMOD_LIBRARIES} BALProblem ParseCmd)
