/**
 * @file    triangulation.cpp
 * @version v1.0.0
 * @date    Oct,27 2017
 * @author  jacob.lin
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

/* find the two photo feature matches points */
void find_feature_matches(const cv::Mat&, const cv::Mat&, std::vector<cv::KeyPoint>&, std::vector<cv::KeyPoint>&, std::vector<cv::DMatch>&);

/* Converts the pixel coordinate system to the normalized imaging plane coordinate system */
cv::Point2d pixel2cam(const cv::Point2d&, const cv::Mat&);

/* find the two photo feature matches points */
void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch> matches, cv::Mat& R, cv::Mat& t);

/* triangulation */
void triangulation(const std::vector<cv::KeyPoint>&, 
    const std::vector<cv::KeyPoint>&, 
    const std::vector<cv::DMatch>&, 
    const cv::Mat&, const cv::Mat&, 
    std::vector<cv::Point3d>& );


int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "usage: triangulation img1 img2." << endl;
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
    
    //-- 3rd, Estimate the movement between two images
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
    
//     //-- verify E=t^R*scale
//     cv::Mat t_x = (cv::Mat_<double>(3,3) << 
//         0,                      -t.at<double>(2, 0),    t.at<double>(1, 0),
//         t.at<double>(2, 0),     0,                      -t.at<double>(0, 0),
//         -t.at<double>(1, 0),    t.at<double>(0, 0),     0 ); 
//     
//     cout << "t^R = " << endl << t_x*R << endl;
//     
//     //-- 验证对极约束
//     // 相机内参,TUM Freiburg2
//     cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
//     for (cv::DMatch m: matches) {
//         cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
//         cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
//         cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
//         cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
//         cv::Mat d = y2.t()*t_x*R*y1;
//         cout << "epipolar constraint = " << d << endl;
//     }
    
    //-- 三角化
    std::vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);
    
    //-- 验证三角化点与特征点的重投影关系
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (int i = 0; i < matches.size(); i ++) {
        cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::Point2d pt1_cam_3d(points[i].x / points[i].z, points[i].y / points[i].z);
        
        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point projected from 3D " << pt1_cam_3d << ", d = " << points[i].z << endl;
        
        // 第二张图
        cv::Point2d pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        
        cout << "point in the second camera frame : " << pt2_cam << endl;
        cout << "point projected from 2nd frame : " << pt2_trans.t() << endl;
    }

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

/**
 * Estimation camera pose
 */
void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch> matches, cv::Mat& R, cv::Mat& t)
{
    // 相机内参,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    
    // 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    
    for (int i = 0; i < matches.size(); i ++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    
    //-- calculates Fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental matrix is " << endl << fundamental_matrix << endl;
    
    //-- calculates Essential matrix
    cv::Mat essential_matrix;
    
    cv::Point2d principal_point(325.1, 249.7);      // 相机光心, TUM dataset标定值
    int focal_length = 521;                         // 相机焦距, TUM dataset标定值
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
//     essential_matrix = cv::findEssentialMat(points1, points2, K, cv::RANSAC);
    
    cout << "essential matrix is " << endl << essential_matrix << endl;
    
    //-- calulates homography matrix
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3, cv::noArray(), 2000, 0.99);
    cout << "homography matrix is " << endl << homography_matrix << endl;
    
    //-- 从本质矩阵中恢复旋转和平移信息
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
//     cv::recoverPose(essential_matrix, points1, points2, K, R, t);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}


/**
 * 
 */
void triangulation(
    const std::vector<cv::KeyPoint>& keypoints_1, 
    const std::vector<cv::KeyPoint>& keypoints_2, 
    const std::vector<cv::DMatch>& matches, 
    const cv::Mat& R, const cv::Mat& t, 
    std::vector<cv::Point3d>& points )
{
    cv::Mat T1 = (cv::Mat_<float> (3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    );
    
    cv::Mat T2 = (cv::Mat_<float> (3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point2f> pts_1, pts_2;
    
    for (cv::DMatch m:matches) {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
    }
    
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
    
    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i ++) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);         // 归一化
        cv::Point3d p (x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}



















