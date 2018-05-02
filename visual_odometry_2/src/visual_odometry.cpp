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
    _matcher_flann(new cv::flann::LshIndexParams(5, 10 ,2)),
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
    _map_point_erase_ratio  = Config::get<double>("map_point_erase_ratio");
    
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
        
        // Extract features from first frame
        extractKeyPonts();
        computeDescriptors();
        
        // the first frame is a key-frame
        addKeyFrame();
        break;
    }
    case OK:
    {
        _curr = frame;
        _curr->_T_c_w = _ref->_T_c_w;
        extractKeyPonts();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        
        /* a good estimation */
        if (checkEstimatedPose() == true)
        {
            _curr->_T_c_w = _T_c_r_estimated;
            optimizeMap();
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
    boost::timer timer;
    _orb->detect(_curr->_color, _keypoints_curr);
    cout << "extract keypoints cost time: " << timer.elapsed() << endl;
}

void VisualOdometry::computeDescriptors(void)
{
    boost::timer timer;
    _orb->compute(_curr->_color, _keypoints_curr, _descriptors_curr);
    cout << "descriptors computation cost time: " << timer.elapsed() << endl;
}

void VisualOdometry::featureMatching(void)
{
    boost::timer timer;
    std::vector<cv::DMatch> matches;
    // select the candidates in map
    cv::Mat desp_map;
    std::vector<MapPoint::Ptr> candidate;
    
    for (auto& allpoints : _map->_map_points)
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image
        if (_curr->isInFrame(p->_pos))
        {
            // add to candidate
            p->_visible_times ++;
            candidate.push_back(p);
            desp_map.push_back(p->_descriptor);
        }
    }
    
    _matcher_flann.match(desp_map, _descriptors_curr, matches);
    // select the best matches
    float min_dis = std::min_element(
                        matches.begin(), matches.end(), 
                        [](const cv::DMatch& m1, const cv::DMatch& m2) 
    { 
        return m1.distance < m2.distance;
    })->distance;
    
    _match_3dpts.clear();
    _match_2dpk_index.clear();
    for (cv::DMatch& m : matches)
    {
        if (m.distance < max<float>(min_dis * _match_ratio, 30.0f))
        {
            _match_3dpts.push_back(candidate[m.queryIdx]);
            _match_2dpk_index.push_back(m.trainIdx);
        }
    }
    
    cout << "good matches: " << _match_3dpts.size() << endl;
    cout << "match cost time: " << timer.elapsed() << endl;    
}

void VisualOdometry::poseEstimationPnP(void)
{
    // construct the 3d 2d observations
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    
    for (int index : _match_2dpk_index)
    {
        pts2d.push_back(_keypoints_curr[index].pt);
    }
    for (MapPoint::Ptr pt : _match_3dpts)
    {
        pts3d.push_back(pt->getPositionCV());
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
        
        // set the inlier map points
        _match_3dpts[index]->_matched_times ++;
    }
    
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    _T_c_r_estimated = Sophus::SE3(
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
    cout << "_T_c_w_estimatied : " << endl << _T_c_r_estimated.matrix() << endl;
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
    Sophus::SE3 T_r_c = _ref->_T_c_w * _T_c_r_estimated.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if (d.norm() > 5.0)
    {
        cout << "reject because motion is too large: " << d.norm() << endl;
        return false;
    }
    
    return true;
    
}

bool VisualOdometry::checkKeyFrame(void)
{
    Sophus::SE3 T_r_c = _ref->_T_c_w * _T_c_r_estimated.inverse();
    Sophus::Vector6d d = T_r_c.log();
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
    if (_map->_keyframes.empty())
    {
        // first key-frame, add all 3d points into map
        for (size_t i = 0; i < _keypoints_curr.size(); i ++)
        {
            double d = _curr->findDepth(_keypoints_curr[i]);
            if (d < 0)
            {
                continue;
            }
            Eigen::Vector3d p_world = _ref->_camera->pixel2world(Eigen::Vector2d(_keypoints_curr[i].pt.x, _keypoints_curr[i].pt.y), _curr->_T_c_w, d);
            Eigen::Vector3d n = p_world - _ref->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(p_world, n, _descriptors_curr.row(i).clone(), _curr.get());
            _map->insertMapPoint(map_point);
        }
    }
    _map->insertKeyFrame(_curr);
    _ref = _curr;
}

void VisualOdometry::addMapPoints(void)
{
    // add the new map points into map
    std::vector<bool> matched(_keypoints_curr.size(), false);
    for (int index : _match_2dpk_index)
    {
        matched[index] = true;
    }
    for (int i = 0; i < _keypoints_curr.size(); i ++)
    {
        if (matched[i] == true)
        {
            continue;
        }
        double d = _ref->findDepth(_keypoints_curr[i]);
        if (d < 0)
        {
            continue;
        }
        Eigen::Vector3d p_world = _ref->_camera->pixel2world(Eigen::Vector2d(_keypoints_curr[i].pt.x, _keypoints_curr[i].pt.y), _curr->_T_c_w, d);
        Eigen::Vector3d n = p_world - _ref->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(p_world, n, _descriptors_curr.row(i).clone(), _curr.get());
        _map->insertMapPoint(map_point);
    }
}

void VisualOdometry::optimizeMap(void)
{
    // remove the hardly seen and no visible points
    for (auto iter = _map->_map_points.begin(); iter != _map->_map_points.end(); )
    {
        if (!_curr->isInFrame(iter->second->_pos))
        {
            iter = _map->_map_points.erase(iter);
            continue;
        }
        
        float match_ratio = float(iter->second->_matched_times) / iter->second->_visible_times;
        if (match_ratio < _map_point_erase_ratio)
        {
            iter = _map->_map_points.erase(iter);
            continue;
        }
        
        double angle = getViewAngle(_curr, iter->second);
        if (angle > M_PI/6.f)
        {
            iter = _map->_map_points.erase(iter);
            continue;
        }
        if (iter->second->_good == false)
        {
            // TODO try triangulate this map point 
        }
        
        iter ++;
    }
    
    if (_match_2dpk_index.size() < 100)
    {
        addMapPoints();
    }

    if (_map->_map_points.size() > 1000)
    {
        // TODO map is too large, remove some one
        _map_point_erase_ratio += 0.05;
    }
    else
    {
        _map_point_erase_ratio = 0.1;
    }
    
    cout << "map points: " << _map->_map_points.size() << endl;
}

double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
{
    Eigen::Vector3d n = point->_pos - frame->getCamCenter();
    n.normalize();
    return acos(n.transpose() * point->_norm);
}

}








