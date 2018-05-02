/**
 * @file    pose_estimation_3d2d.cpp
 * @version v1.0.0
 * @date    Oct,27 2017
 * @author  jacob.lin
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <chrono>

using namespace std;
using namespace cv;

/* find the two photo feature matches points */
void find_feature_matches(const cv::Mat&, const cv::Mat&, std::vector<cv::KeyPoint>&, std::vector<cv::KeyPoint>&, std::vector<cv::DMatch>&);

/* Converts the pixel coordinate system to the normalized imaging plane coordinate system */
cv::Point2d pixel2cam(const cv::Point2d&, const cv::Mat&);

/*  */
void bundleAdjustment( const std::vector<cv::Point3f>, const std::vector<cv::Point2f>, const cv::Mat&, cv::Mat&, cv::Mat&);

int main(int argc, char** argv)
{
    if (argc != 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2." << endl;
        return 1;
    }
    
    //-- 1st, read photo
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if (img_1.empty() || img_2.empty()) {
        cout << "img1 or img2 is no find." << endl;
        return 1;
    }
    
    //-- 2nd, find the two photo feature matches points
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "feature matches points total is " << matches.size() << endl;
    
    //-- 3rd, create 3 dimensions point correspondences
    cv::Mat d1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);      // 深度图为16位无符号数，单通道图像
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int (keypoints_1[m.queryIdx].pt.y))[int (keypoints_1[m.queryIdx].pt.x)];
        
        if (d == 0) {   /* bad depth */
            continue;
        }
        float dd = d/5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3d (p1.x*dd, p1.y*dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    
    //-- 4th, use PnP or bundle adjustment compute camera pose.
    cv::Mat r, t;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);    // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R;
    cv::Rodrigues(r, R);    // r为旋转向量形式，用Rodrigues公式转换为矩阵
    
    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;
    
    cout << "calling bundle adjustment" << endl;
    
    bundleAdjustment(pts_3d, pts_2d, K, R, t);
    
    return 0;
}

/**
 * find the two photo feature matches points
 */
void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches)
{
    //-- Initialize
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    
    //-- 1st: detect Oriented FAST KeyPoint 
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    
    //-- 2nd: Calculates BRIEF descriptors with KeyPoint's coordinate
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    
    //-- 3rd: 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    
    //-- 4th: 匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    
    /* 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离 */
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    printf("-- Max distance : %f \r\n", max_dist);
    printf("-- Min distance : %f \r\n", min_dist);
    
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_2.rows; i ++) {
        if (match[i].distance <= max(2*min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

/**
 * Converts the pixel coordinate system to the normalized imaging plane coordinate system
 */
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2d
        (
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
}

void bundleAdjustment(
    const std::vector<cv::Point3f> points_3d, 
    const std::vector<cv::Point2f> points_2d,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t )
{
    // Initialize g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;   // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linear_solver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block((std::unique_ptr<Block::LinearSolverType>)(linear_solver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg((std::unique_ptr<Block>)solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    // vertex 
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();    // camera pose
    Eigen::Matrix3d R_mat;
    R_mat << \
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), \
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), \
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(
        g2o::SE3Quat(
            R_mat, 
            Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))
        )
    );
    optimizer.addVertex(pose);
    
    int index = 1;
    for (const cv::Point3f p:points_3d) {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index ++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);   // g2o 中必须设置 marg 
        optimizer.addVertex(point);
    }
    
    // parameter : camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0
    );
    camera->setId(0);
    optimizer.addParameter(camera);
    
    // edges
    index = 1;
    for (const cv::Point2f p:points_2d) {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    
    cout << endl << "after optimization: " << endl;
    cout << "T = " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}









