/**
 * @file    feature_extraction.cpp
 * @version v1.0.0
 * @date    Oct,11 2017
 * @author  Jacob.lin
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }
    
    //-- read photos
    cv::Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    //-- Initialize
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create ( "BruteForce-Hamming" );
    
    //-- frist: detect Oriented FAST Key-point
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    
    //-- second: Calculates BRIEF descriptors with Key-point's coordinate
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    
    cv::Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB Feature-point", outimg1);
    
    //-- third: 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    
    //-- fourth: 匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    
    /* 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离 */
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    printf("-- Max distance : %f \n", max_dist);
    printf("-- Min distance : %f \n", min_dist);
    
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2*min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }
    
    //-- fifth: 绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("所有匹配点对", img_match);
    imshow("优化后匹配点对", img_goodmatch);
    waitKey(0);
    
    return 0;
}









