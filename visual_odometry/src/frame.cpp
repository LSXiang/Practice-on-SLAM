/**
 * @file    frame.cpp
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#include "frame.h"

namespace visual_odometry
{

Frame::Frame() : _id(-1), _time_stamp(-1), _camera(nullptr)
{
}

Frame::Frame(long id, double time_stamp, Sophus::SE3 T_c_w, Camera::Ptr camera, cv::Mat color, cv::Mat depth) : \
    _id(id),                    \
    _time_stamp(time_stamp),    \
    _T_c_w(T_c_w),              \
    _camera(camera),            \
    _color(color),              \
    _depth(depth)
{
}

Frame::~Frame()
{
}

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;
    return Frame::Ptr(new Frame(factory_id ++));
}

double Frame::findDepth(const cv::KeyPoint& kp)
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = _depth.ptr<ushort>(y)[x];
    if (d != 0)
    {
        return double(d) / _camera->depth_scale_;
    }
    else
    {
        /* check the nearby points */
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, -1, 0, 1};
        for (int i = 0; i < 4; i ++)
        {
            d = _depth.ptr<ushort>(y + dy[i])[x + dx[i]];
            if (d != 0)
            {
                return double(d) / _camera->depth_scale_;
            }
        }
    }
    return -1.0;
}

Eigen::Vector3d Frame::getCamCenter() const
{
    return _T_c_w.inverse().translation();
}

bool Frame::isInFrame(const Eigen::Vector3d& pt_world)
{
    Eigen::Vector3d p_cam = _camera->world2camera(pt_world, _T_c_w);
    if (p_cam(2, 0) < 0)
    {
        return false;
    }
    Eigen::Vector2d pixel = _camera->world2pixel(pt_world, _T_c_w);
    
    return pixel(0, 0) > 0 && pixel(1, 0) > 0 && pixel(0, 0) < _color.cols && pixel(1, 0) < _color.rows;
}

}



