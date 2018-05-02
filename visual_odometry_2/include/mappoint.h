/**
 * @file    mappoint.h
 * @version v1.1.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 * 
 * @note    ver1.1.0(Dec,22 2017) : add map point storage mechanism to create local-map perfect visual odometry
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
    unsigned long                       _id;            // ID
    static unsigned long                _factory_id;    // factory ID
    bool                                _good;          // whether a good point
    Eigen::Vector3d                     _pos;           // Position in world
    Eigen::Vector3d                     _norm;          // Normal of viewing direction
    cv::Mat                             _descriptor;    // Descriptor for matching
    
    std::list<visual_odometry::Frame*>  _observed_frames;   // key-points that can observe this point
    
    int                                 _matched_times;     // being an inliner in pose estimation
    int                                 _visible_times;     // being visible in current frame
    
    MapPoint(void);
    MapPoint(long int id, Eigen::Vector3d position, Eigen::Vector3d norm, visual_odometry::Frame* frame = nullptr, const cv::Mat& descriptor = cv::Mat());
    
    inline cv::Point3f getPositionCV(void) const
    {
        return cv::Point3f(_pos(0, 0), _pos(1, 0), _pos(2, 0));
    }
    
    /* factory function */
    static MapPoint::Ptr createMapPoint(void);
    static MapPoint::Ptr createMapPoint(const Eigen::Vector3d& pos_world, const Eigen::Vector3d& normal, const cv::Mat& descriptor, visual_odometry::Frame* frame);
};

}

#endif /* __MAPPOINT_H__ */
