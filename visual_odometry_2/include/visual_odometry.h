/**
 * @file    visual_odometry.h
 * @version v1.0.0
 * @date    Dec,13 2017
 * @author  jacob.lin
 */

#ifndef __VISUAL_ODOMETRY__
#define __VISUAL_ODOMETRY__

#include "common_include.h"
#include "map.h"

#include <opencv2/features2d/features2d.hpp>

namespace visual_odometry
{

class VisualOdometry 
{
public:
    typedef std::shared_ptr<VisualOdometry> Ptr;
    enum VOState
    {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };
    
    VOState     _state;     // current VO status
    Map::Ptr    _map;       // map with all frames and map points
    Frame::Ptr  _ref;       // reference frame
    Frame::Ptr  _curr;      // current frame
    
    cv::Ptr<cv::ORB> _orb;  // orb detector and computer
    std::vector<cv::KeyPoint>   _keypoints_curr;    // keypoints in current frame
    cv::Mat                     _descriptors_curr;  // descriptor in current frame
    
    cv::FlannBasedMatcher       _matcher_flann;     // flann matcher
    std::vector<MapPoint::Ptr>  _match_3dpts;       // matched 3d points
    std::vector<int>            _match_2dpk_index;  // matched 2d pixels (index of kp_curr)
    
    
    Sophus::SE3 _T_c_r_estimated;   // the estimated pose of current frame
    int         _num_inliers;       // number of inlier features in icp
    int         _num_lost;          // number of lost times
    
    // parameters
    int     _num_of_features;       // number of features
    double  _scale_factor;          // scale in image pyramid
    int     _level_pyramid;         // number of pyramid levels
    float   _match_ratio;           // ratio for selecting good matches 
    int     _max_num_lost;          // max number of continuous lost times
    int     _min_inliers;           // mininum inliers
    
    double  _key_frame_min_rot;     // minimal rotation of two key-frames
    double  _key_frame_min_trans;   // minimal translation of two key-frames
    
    double _map_point_erase_ratio;   // remove map point ratio
    
public:
    // functions
    VisualOdometry(void);
    ~VisualOdometry();
    
    bool addFrame(Frame::Ptr frame);    // add a new frame
    
protected:
    // inner operation
    void extractKeyPonts(void);
    void computeDescriptors(void);
    void featureMatching(void);
    void poseEstimationPnP(void);
    void optimizeMap(void);
    
    void addKeyFrame(void);
    void addMapPoints(void);
    bool checkEstimatedPose(void);
    bool checkKeyFrame(void);
    
    double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
};

}



#endif /* __VISUAL_ODOMETRY__ */














