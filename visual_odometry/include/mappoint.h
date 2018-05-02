/**
 * @file    mappoint.h
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#ifndef __MAPPOINT_H__
#define __MAPPOINT_H__

#include "common_include.h"

namespace visual_odometry
{

/* forward declare */
class Frame;

class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint>   Ptr;
    unsigned long                       _id;        // ID
    Eigen::Vector3d                     _pos;       // Position in world
    Eigen::Vector3d                     _norm;      // Normal of viewing direction
    cv::Mat                             _descriptor;// Descriptor for matching
    int                                 _observed_times;    // being observed by feature matching algo.
    int                                 _correct_times;     // being an inliner in pose estimation
    
    MapPoint();
    MapPoint(long id, Eigen::Vector3d position, Eigen::Vector3d norm);
    
    /* factory function */
    static MapPoint::Ptr createMapPoint();
};

}

#endif /* __MAPPOINT_H__ */
