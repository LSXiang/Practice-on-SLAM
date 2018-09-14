##
# Copyright (c) 2018, The Akatsuki(Jacob.lsx). All rights reserved.
#
# custombundle_ceres/CMakeLists.txt
#
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} bal_ceres_src)

add_executable(customBundle_Ceres ${bal_ceres_src})

target_link_libraries(customBundle_Ceres ${CERES_LIBRARIES} BALProblem ParseCmd)
