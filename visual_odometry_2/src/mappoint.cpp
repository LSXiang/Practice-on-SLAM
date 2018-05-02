/**
 * @file    mappoint.cpp
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 * 
 * @note    ver1.1.0(Dec,22 2017) : add map point storage mechanism to create local-map perfect visual odometry
 */

#include "mappoint.h"

namespace visual_odometry
{

unsigned long MapPoint::_factory_id = 0;

MapPoint::MapPoint(void) :
    _id(-1),
    _good(true),
    _pos(Eigen::Vector3d(0, 0, 0)),
    _norm(Eigen::Vector3d(0, 0, 0)),
    _matched_times(0),
    _visible_times(0)
    
{
}

MapPoint::MapPoint(long id, Eigen::Vector3d position, Eigen::Vector3d norm, visual_odometry::Frame* frame, const cv::Mat& descriptor) :
    _id(id),
    _good(true),
    _pos(position),
    _norm(norm),
    _descriptor(descriptor),
    _matched_times(1),
    _visible_times(1)
    
{
    _observed_frames.push_back(frame);
}


MapPoint::Ptr MapPoint::createMapPoint(void)
{
    return MapPoint::Ptr(new MapPoint(_factory_id ++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
}

MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d& pos_world, const Eigen::Vector3d& normal, const cv::Mat& descriptor, visual_odometry::Frame* frame)
{
    return MapPoint::Ptr(new MapPoint(_factory_id ++, pos_world, normal, frame, descriptor));
}

}

