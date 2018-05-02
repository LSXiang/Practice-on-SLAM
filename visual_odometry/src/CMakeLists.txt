# create VO library 
add_library(visual_odometry_slam SHARED STATIC
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry.cpp
)

target_link_libraries(visual_odometry_slam ${VisualOdometryLibraries})
