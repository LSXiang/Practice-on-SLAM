/**
 * @file    commom_include.h
 * @version v1.0.0
 * @date    Dec,11 2017
 * @author  jacob.lin
 */

#ifndef __COMMON_INCLUDE_H__
#define __COMMON_INCLUDE_H__

/* define the commomly included file to avoid a long include list */

// -- for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

// -- for Sophus
#include <sophus/se3.h>
using Sophus::SE3;

// -- for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

// -- std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>

using namespace std;

#endif  /* __COMMON_INCLUDE_H__ */
