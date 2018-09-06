/**
 * Copyright (c) 2018, The Akatsuki(Jacob.lsx). All rights reserved.
 */


#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <chrono>

using namespace std;
using namespace cv;

/* find the two photo feature matches points */
void find_feature_matches(const cv::Mat&, const cv::Mat&, std::vector<cv::KeyPoint>&, std::vector<cv::KeyPoint>&, std::vector<cv::DMatch>&);

/* Converts the pixel coordinate system to the normalized imaging plane coordinate system */
cv::Point2d pixel2cam(const cv::Point2d&, const cv::Mat&);

/*  */
void pose_estimation_3d3d(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&, cv::Mat&, cv::Mat&);

/*  */
void bundleAdjustment(const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&, cv::Mat&, cv::Mat&);

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2." << endl;
        return 1;
    }
    
    //-- 1st, read photo
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    if (img_1.empty() || img_2.empty()) {
        cerr << "img1 or img2 is empty." << endl;
        return 1;
    }
    
    //-- 2nd, find the two photo feature matches points
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "feature matches points total is " << matches.size() << endl;
    
    //-- 3rd, create 3 dimensions point correspondences
    cv::Mat depth1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);      // 深度图为16位无符号数，单通道图像
    cv::Mat depth2 = cv::imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    if (depth1.empty() || depth2.empty()) {
        cerr << "depth1 or depth2 is empty." << endl;
        return 1;
    }
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts1, pts2;
    
    for (cv::DMatch m : matches) {
        ushort d1 = depth1.ptr<unsigned short>((int)keypoints_1[m.queryIdx].pt.y)[(int)keypoints_1[m.queryIdx].pt.x];
        ushort d2 = depth2.ptr<unsigned short>((int)keypoints_2[m.trainIdx].pt.y)[(int)keypoints_2[m.trainIdx].pt.x];
        if (d1 == 0 || d2 == 0) {
            continue;
        }
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }
    cout << "3d-3d pairs: " << pts1.size() << endl;
    
    cv::Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP vid SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;
    
    cout << "calling bundle adjustment. \r\n";
    
    bundleAdjustment(pts1, pts2, R, t);
    
    // verify p1 = R*p2 + t
    for (int i = 0; i < 5; i ++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "R*p2 + t = " << R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << endl;
        cout << endl;
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

void pose_estimation_3d3d(const std::vector<cv::Point3f>& pts1, const std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t)
{
    // center of mass
    cv::Point3f p1, p2;
    int N = pts1.size();
    for (int i = 0; i < N; i ++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(Vec3f(p1) / N);
    p2 = cv::Point3f(Vec3f(p2) / N);
    
    // remove the center
    std::vector<cv::Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i ++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i ++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W = " << W << endl;
    
    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 利用SVD求解3D-3D变换，需要U和V的行列式同号。换言之，旋转矩阵的行列式只能为1，不能为-1
    if (U.determinant() * V.determinant() < 0) {
        for (int x = 0; x < 3; ++x) {
            U(x, 2) *= -1;
        }
    }
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;
    
    
    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
    
    // convert to cv::mat
    R = (cv::Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

class ReprojectionError {
public:
    ReprojectionError(const cv::Point3f &point1, const cv::Point3f &point2) : pt1_(point1), pt2_(point2) {}
    
    template<typename T>
    bool operator() (const T* const r, const T* const t, T* residuals) const
    {
        T p[3];
        T pt2[3] = {(T)pt2_.x, (T)pt2_.y, (T)pt2_.z};
        
        ceres::AngleAxisRotatePoint(r, pt2, p);
                
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];
        
        residuals[0] = p[0] - (T)pt1_.x;
        residuals[1] = p[1] - (T)pt1_.y;
        residuals[2] = p[2] - (T)pt1_.z;
        
        return true;
    }
    
    static ceres::CostFunction* create(const cv::Point3f &point1, const cv::Point3f &point2)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 3, 3, 3>(new ReprojectionError(point1, point2));
    }
    
private:
    cv::Point3f pt1_, pt2_;
};

void bundleAdjustment(const std::vector<cv::Point3f>& pts1, const std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t)
{
    cv::Mat r;
    cv::Rodrigues(R, r);
    
    double rotation[3] = {0.f};
    double translation[3] = {0.f};
    
#if 1
    for (int i = 0; i < 3; i++) {
        rotation[i] = r.at<double>(i, 0);
        translation[i] = r.at<double>(i, 0);
    }
#endif
    
    ceres::Problem problem;
    for (int i = 0; i < pts1.size(); i ++) {
        ceres::CostFunction *costFunction = ReprojectionError::create(pts1[i], pts2[i]);
        problem.AddResidualBlock(costFunction, nullptr, rotation, translation);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    ceres::Solve(options, &problem, &summary);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    
    cout << summary.BriefReport() << endl;
    
    cout << "optimization costs time: " << time_used.count() << " seconds." << "\r\n";
    
    cout << endl << "after optimization: " << endl;
    
    cv::Mat R_vec = (cv::Mat_<double>(3, 1) << rotation[0], rotation[1], rotation[2]);
    cv::Mat Rotation;
    cv::Rodrigues(R_vec, Rotation);
    
    cout << "R = \r\n" << Rotation << endl;
    cout << "t = \r\n" << translation[0] << ", " << translation[1] << ", " << translation[2] << endl;
    
    R = Rotation;
    t = (cv::Mat_<double>(3,1) << translation[0], translation[1], translation[2]);
}



