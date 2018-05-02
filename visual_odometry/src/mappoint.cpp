/**
 * @file    mappoint.cpp
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#include "mappoint.h"

namespace visual_odometry
{

MapPoint::MapPoint() :
    _id(-1),
    _pos(Eigen::Vector3d(0, 0, 0)),
    _norm(Eigen::Vector3d(0, 0, 0)),
    _observed_times(0),
    _correct_times(0)
{
}

MapPoint::MapPoint(long id, Eigen::Vector3d position, Eigen::Vector3d norm) :
    _id(id),
    _pos(position),
    _norm(norm),
    _observed_times(0),
    _correct_times(0)
{
}


MapPoint::Ptr MapPoint::createMapPoint()
{
    static long factory_id = 0;
    return MapPoint::Ptr(new MapPoint(factory_id ++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
}


}

