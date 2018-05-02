/**
 * @file    map.cpp
 * @version v1.0.0
 * @date    Dec,13 2017
 * @author  jacob.lin
 */

#include "map.h"

namespace visual_odometry
{

void Map::insertKeyFrame(Frame::Ptr frame)
{
    std::cout << "Key frame size = " << _keyframes.size() << endl;
    if (_keyframes.find(frame->_id) == _keyframes.end())
    {
        _keyframes.insert(make_pair(frame->_id, frame));
    }
    else
    {
        _keyframes[frame->_id] = frame;
    }
}

void Map::insertMapPoint(MapPoint::Ptr map_point)
{
    if (_map_points.find(map_point->_id) == _map_points.end())
    {
        _map_points.insert(make_pair(map_point->_id, map_point));
    }
    else 
    {
        _map_points[map_point->_id] = map_point;
    }
}

}
