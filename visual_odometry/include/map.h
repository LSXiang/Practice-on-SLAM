/**
 * @file    map.h
 * @version v1.0.0
 * @date    Dec,13 2017
 * @author  jacob.lin
 */

#ifndef __MAP_H__
#define __MAP_H__

#include "common_include.h"
#include "frame.h"
#include "mappoint.h"

namespace visual_odometry
{

class Map 
{
public:
    typedef std::shared_ptr<Map>    Ptr;
    std::unordered_map<unsigned long, MapPoint::Ptr>    _map_points;    // all landmarks
    std::unordered_map<unsigned long, Frame::Ptr>       _keyframes;     // all key-frames
    
    Map() {}
    
    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);
};

}


#endif /* __MAP_H__ */
