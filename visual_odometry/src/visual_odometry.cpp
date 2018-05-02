/**
 * @file    visual_odometry.cpp
 * @version v1.0,0
 * @date    Dec,13 2017
 * @author  jacob.lin
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "config.h"
#include "visual_odometry.h"
#include "g2o_types.h"

namespace visual_odometry
{

VisualOdometry::VisualOdometry(void) :
    _state(INITIALIZING),
    _map(new Map),
    _ref(nullptr),
    _curr(nullptr),
    _num_inliers(0),
    _num_lost(0)
{
    _num_of_features        = Config::get<int>("number_of_features");
    _scale_factor           = Config::get<double>("scale_factor");
    _level_pyramid          = Config::get<int>("level_pyramid");
    _match_ratio            = Config::get<float>("match_ratio");
    _max_num_lost           = Config::get<float>("max_num_lost");
    _min_inliers            = Config::get<int>("min_inliers");
    _key_frame_min_rot      = Config::get<double>("keyframe_rotation");
    _key_frame_min_trans    = Config::get<double>("keyframe_translation");
    
    _orb = cv::ORB::create(_num_of_features, _scale_factor, _level_pyramid);
}

VisualOdometry::~VisualOdometry(void)
{
}

bool VisualOdometry::addFrame(Frame::Ptr frame)
{
    switch (_state)
    {
    case INITIALIZING:
    {
        _state = OK;
        _curr = _ref = frame;
        _map->insertKeyFrame(frame);
        
        // Extract features from first frame
        extractKeyPonts();
        computeDescriptors();
        
        // Compute the 3d position of features in ref frame
        setRef3DPoints();
        break;
    }
    case OK:
    {
        _curr = frame;
        extractKeyPonts();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        
        /* a good estimation */
        if (checkEstimatedPose() == true)
        {
            _curr->_T_c_w = _T_c_r_estimated * _ref->_T_c_w;    // T_c_w = T_c_r * T_r_w
            _ref = _curr;
            setRef3DPoints();
            _num_lost = 0;
            if (checkKeyFrame() == true)        // is a key-frame
            {
                addKeyFrame();
            }
        }
        else    // bad estiamation due to various reasons
        {
            _num_lost ++;
            if (_num_lost > _max_num_lost)
            {
                _state = LOST;
            }
        }
        
        break;
    }
    case LOST:
    {
        std::cout << "VO has lost." << endl;
        break;
    }
    }
    
    return true;
}

void VisualOdometry::extractKeyPonts(void)
{
    _orb->detect(_curr->_color, _keypoints_curr);
}

void VisualOdometry::computeDescriptors(void)
{
    _orb->compute(_curr->_color, _keypoints_curr, _descriptors_curr);
}

void VisualOdometry::featureMatching(void)
{
    // match desp_ref and desp_curr, use OpenCV's brute force match
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(_descriptors_ref, _descriptors_curr, matches);
    
    // select the best matches
    float min_dis = std::min_element(
                        matches.begin(), matches.end(), 
                        [](const cv::DMatch& m1, const cv::DMatch& m2) 
    { 
        return m1.distance < m2.distance;
    })->distance;
    
    _feature_matches.clear();
    for (cv::DMatch& m : matches)
    {
        if (m.distance < max<float>(min_dis * _match_ratio, 30.0))
        {
            _feature_matches.push_back(m);
        }
    }
    cout << "good matches: " << _feature_matches.size() << endl;
}

void VisualOdometry::setRef3DPoints(void)
{
    // select the features with depth measurements
    _pts_3d_ref.clear();
    _descriptors_ref = cv::Mat();
    for (size_t i = 0; i < _keypoints_curr.size(); i++) 
    {
        double d = _ref->findDepth(_keypoints_curr[i]);
        if (d > 0)
        {
            Eigen::Vector3d p_cam = _ref->_camera->pixel2camera(Eigen::Vector2d(_keypoints_curr[i].pt.x, _keypoints_curr[i].pt.y), d);
            _pts_3d_ref.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
            _descriptors_ref.push_back(_descriptors_curr.row(i));
        }
    }
}

void VisualOdometry::poseEstimationPnP(void)
{
    // construct the 3d 2d observations
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    
    for (cv::DMatch m : _feature_matches)
    {
        pts3d.push_back(_pts_3d_ref[m.queryIdx]);
        pts2d.push_back(_keypoints_curr[m.trainIdx].pt);
    }
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        _ref->_camera->fx_,         0,          _ref->_camera->cx_,
                0,          _ref->_camera->fy_, _ref->_camera->cy_,
                0,                  0,                  1
    );
    
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
    _num_inliers = inliers.rows;
    cout << "pnp inliers: " << _num_inliers << endl;
    _T_c_r_estimated = Sophus::SE3(
        Sophus::SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
        Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
    );
    
    /* using bundle adjustment to optimize the pose */
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block((std::unique_ptr<Block::LinearSolverType>) linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg((std::unique_ptr<Block>)solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            _T_c_r_estimated.rotation_matrix(),
            _T_c_r_estimated.translation()
        )
    );
    optimizer.addVertex(pose);
    
    // edges
    for (int i = 0; i < inliers.rows; i ++)
    {
        int index = inliers.at<int>(i, 0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->_camera = _curr->_camera.get();
        edge->_point = Eigen::Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
        edge->setMeasurement(Eigen::Vector2d(pts2d[index].x, pts2d[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    _T_c_r_estimated = Sophus::SE3(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
}

bool VisualOdometry::checkEstimatedPose(void)
{
    // check if the estimated pose is good
    if (_num_inliers < _min_inliers)
    {
        cout << "reject because inlier is too small: " << _num_inliers << endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = _T_c_r_estimated.log();
    if (d.norm() > 5.0)
    {
        cout << "reject because motion is too large: " << d.norm() << endl;
        return false;
    }
    return true;
    
}

bool VisualOdometry::checkKeyFrame(void)
{
    Sophus::Vector6d d = _T_c_r_estimated.log();
    Eigen::Vector3d trans = d.head<3>();
    Eigen::Vector3d rot = d.tail<3>();
    if (rot.norm() > _key_frame_min_rot || trans.norm() > _key_frame_min_trans)
    {
        return true;
    }
    return false;
}

void VisualOdometry::addKeyFrame(void)
{
    cout << "adding a key-frame" << endl;
    _map->insertKeyFrame(_curr);
}

}








