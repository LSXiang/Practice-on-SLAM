/**
 * @file    frame.h
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#ifndef __FRAME_H__
#define __FRAME_H__

#include "common_include.h"
#include "camera.h"

namespace visual_odometry
{

/* forward declare */
class MapPoint;

class Frame
{
public:     
    /* data members */
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long                  _id;         // id of this frame
    double                         _time_stamp; // when it is recorded
    Sophus::SE3                    _T_c_w;      // transform from world to camera
    Camera::Ptr                    _camera;     // Pinhole RGBD Camera model
    cv::Mat                        _color, _depth;  // color and depth image
    
public:
    Frame();
    Frame(long id, double _time_stamp = 0, Sophus::SE3 T_c_w = Sophus::SE3(), Camera::Ptr camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());
    ~Frame();
    
    /* factory function */
    static Frame::Ptr createFrame();
    
    /* find the depth in depth map */
    double findDepth(const cv::KeyPoint& kp);
    
    /* Get Camera Center */
    Eigen::Vector3d getCamCenter() const;
    
    /* check if a point is in this frame */
    bool isInFrame(const Eigen::Vector3d& pt_world);
};

}

#endif /* __FRAME_H__ */
