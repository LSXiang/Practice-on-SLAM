/**
 * Copyright (c) 2018, The Akatsuki(Jacob.lsx). All rights reserved.
 */


#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <chrono>

using namespace std;
using namespace cv;

/* find the two photo feature matches points */
void find_feature_matches(const cv::Mat&, const cv::Mat&, std::vector<cv::KeyPoint>&, std::vector<cv::KeyPoint>&, std::vector<cv::DMatch>&);

/* Converts the pixel coordinate system to the normalized imaging plane coordinate system */
cv::Point2d pixel2cam(const cv::Point2d&, const cv::Mat&);

void bundleAdjustment( const std::vector<cv::Point3f>, const std::vector<cv::Point2f>, const cv::Mat&, cv::Mat&, cv::Mat&);

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
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
    
    cout << "\r\ncalling bundle adjustment" << endl;
    
    bundleAdjustment(pts_3d, pts_2d, K, r, t);
    
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

class ReprojectionError
{
public:
    ReprojectionError(const cv::Point2f p_2d, const cv::Point3f p_3d) : p_2d_(p_2d), p_3d_(p_3d) {}
    
    template<typename T>
    bool operator()(const T* const T_, const T* const point_3d_,  T* residual_) const;
    
    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction* create(const cv::Point2f &p_2d, const cv::Point3f &p_3d);
    
private:
    template<typename T>
    static inline void camProjectionWithoutDistortion(const T* const r, const T* const t, const T* const pt_3d, T* predicted);
    
    cv::Point2f p_2d_;
    cv::Point3f p_3d_;
    static float fx_, fy_, cx_, cy_;
};

float ReprojectionError::fx_ = 520.9; 
float ReprojectionError::fy_ = 521.0;
float ReprojectionError::cx_ = 325.1;
float ReprojectionError::cy_ = 249.7;

template<typename T>
bool ReprojectionError::operator()(const T* const r_, const T* const t_,  T* residual_) const
{
    T p[3] = {(T)p_3d_.x, (T)p_3d_.y, (T)p_3d_.z};
    T predicted[2];
    
    // T_[0, 1, 2] are the angle-axis rotation
    // T_[3, 4, 5] are the translation
    camProjectionWithoutDistortion(r_, t_, p, predicted);
    
    // The error is the difference between the predicted and observed position
    residual_[0] = predicted[0] - T(p_2d_.x);
    residual_[1] = predicted[1] - T(p_2d_.y);
    
//     cout << residual_[1] << ", " << residual_[2] << endl;
}

template<typename T>
inline void ReprojectionError::camProjectionWithoutDistortion(const T* const r, const T* const t, const T* const pt_3d, T* predicted)
{
    T p[3];
    ceres::AngleAxisRotatePoint(r, pt_3d, p);
    
    p[0] += t[0];
    p[1] += t[1];
    p[2] += t[2];
    
    T px_normalized = p[0] / p[2];
    T py_normalized = p[1] / p[2];
    
    predicted[0] = (T)fx_ * px_normalized + (T)cx_;
    predicted[1] = (T)fy_ * py_normalized + (T)cy_;
}

ceres::CostFunction* ReprojectionError::create(const cv::Point2f &p_2d, const cv::Point3f &p_3d)
{
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(p_2d, p_3d)));
}

void bundleAdjustment(
    const std::vector<cv::Point3f> points_3d,
    const std::vector<cv::Point2f> points_2d,
    const cv::Mat& K,
    cv::Mat& r,
    cv::Mat& t )
{
    double rotation[3];
    double translation[3];
    for (int i = 0; i < 3; i++) {
        rotation[i] = r.at<double>(i, 0);
        translation[i] = t.at<double>(i, 0);
    }
    
    ceres::Problem problem;
    for (int i = 0; i < points_2d.size(); i ++) {
        ceres::CostFunction* cost_function = ReprojectionError::create(points_2d[i], points_3d[i]);
        problem.AddResidualBlock(cost_function, nullptr, rotation, translation);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    ceres::Solve(options, &problem, &summary);
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    
    cout << summary.BriefReport() << endl;
    
    cout << endl << "after optimization: " << endl;
    
    cv::Mat r_vec = (cv::Mat_<double>(3, 1) << rotation[0], rotation[1], rotation[2]);
    cv::Mat R;
    cv::Rodrigues(r_vec, R);
    
    cout << "R = \r\n" << R << endl;
    cout << "t = \r\n" << translation[0] << ", " << translation[1] << ", " << translation[2] << endl;
    
//     cout << "T = " << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}






