#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <boost/timer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.h>

/* parameters */
const int boarder = 20;         // boundary
const int width = 640;
const int height = 480;

const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;

const int ncc_window_size = 2;  // NCC window width half size
const int ncc_area = (2*ncc_window_size + 1) * (2*ncc_window_size + 1);     // NCC window area size

const double min_cov = 0.1f;    // convergence condition: minimum variance
const double max_cov = 10.f;    // divergence  condition: maximum variance

/* functions declaration */
bool readDatasetFiles(
    const std::string& path, 
    std::vector<std::string>& color_image_files, 
    std::vector<Sophus::SE3>& pose
);

bool update(
    const cv::Mat& ref, 
    const cv::Mat& curr, 
    const Sophus::SE3& T_c_r, 
    cv::Mat depth, 
    cv::Mat& depth_cov
);

bool epipolarSearch(
    const cv::Mat& ref, 
    const cv::Mat& curr, 
    const Sophus::SE3& T_c_r, 
    const Eigen::Vector2d& pt_ref, 
    const double& depth_mu, 
    const double& depth_cov, 
    Eigen::Vector2d& pt_curr
);

bool updateDepthFilter(
    const Eigen::Vector2d& pt_ref,
    const Eigen::Vector2d& pt_curr,
    const Sophus::SE3& T_c_r,
    cv::Mat& depth,
    cv::Mat& depth_cov
);

double computeNCC(
    const cv::Mat& ref,
    const cv::Mat& curr,
    const Eigen::Vector2d& pt_ref,
    const Eigen::Vector2d& pt_curr
);

inline double getBilinearInterpolatedValue(const cv::Mat& image, const Eigen::Vector2d& pt)
{
    uchar* d = & image.data[int(pt.y() * image.step + pt.x())];
    
    double xx = pt.x() - floor(pt.x());
    double yy = pt.y() - floor(pt.y());
    
    return ( (1.f-xx) * (1.f-yy) * double(d[0]) + 
                xx    * (1.f-yy) * double(d[1]) +
             (1.f-xx) *    yy    * double(d[image.step]) +
                xx    *    yy    * double(d[image.step + 1]) ) / 255.0;
}

/* some tools functions */
void plotDepth(const cv::Mat& depth);

inline Eigen::Vector3d px2cam(const Eigen::Vector2d px)
{
    return Eigen::Vector3d(
        (px.x() - cx) / fx,
        (px.y() - cy) / fy,
        1.0
    );
}

inline Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam)
{
    return Eigen::Vector2d(
        p_cam.x() * fx / p_cam.z() + cx,
        p_cam.y() * fy / p_cam.z() + cy
    );
}

inline bool inside(const Eigen::Vector2d& pt)
{
    return (pt.x() >= boarder && pt.y() >= boarder
        &&  pt.x() + boarder <= width && pt.y() + boarder <= height); 
}

void showEpipolarMatch(
    const cv::Mat& ref, 
    const cv::Mat& curr, 
    const Eigen::Vector2d& px_ref, 
    const Eigen::Vector2d& px_curr
);

void showEpipolarLine(
    const cv::Mat& ref,
    const cv::Mat& curr,
    const Eigen::Vector2d& px_ref,
    const Eigen::Vector2d& px_mincurr,
    const Eigen::Vector2d& px_maxcurr
);

int main(int argc, char** argv)
{
    
    
    return 0u;
}

bool readDatasetFiles(const std::string& path, std::vector<std::string>& color_image_files, std::vector<Sophus::SE3>& pose)
{
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;
    
    while(!fin.eof()) {
        /* the form of dataset : image's name, tx, ty, tz, qx, qy, qz, qw. NOTE: the transformation matrix is camera to world */
        std::string image_name;
        fin >> image_name;
        
        double data[7];
        for (double& d : data) fin >> d;
        
        color_image_files.push_back(path + std::string("/images/") + image_name);
        pose.push_back(
            Sophus::SE3(Sophus::Quaterniond(data[6], data[3], data[4], data[5]),
                        Sophus::Vector3d(data[0], data[1], data[2]))
        );
        
        if (!fin.good()) break;
    }
    
    return true;
}

/* update the global depth map */
bool update(const cv::Mat& ref, const cv::Mat& curr, const Sophus::SE3& T_c_r, cv::Mat depth, cv::Mat& depth_cov)
{
#pragma omp parallel for
    for (int x = boarder; x < width - boarder; x++) {
#pragma omp parallel for 
        for (int y = boarder; y < height - boarder; y++) {
            // Traversing every pixel
            if (depth_cov.ptr<double>(y)[x] < min_cov || depth_cov.ptr<double>(y)[x] > max_cov)     // the depth was convergence / divergence
                continue;
            
            // match the pixel in the epipolar between the reference and the current
            Eigen::Vector2d pt_curr;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_c_r,
                Eigen::Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov.ptr<double>(y)[x]),
                pt_curr
            );
            
            if (ret == false)   // match false
                continue;
            
            // 取消该注释以显示匹配
//             showEpipolarMatch(ref, curr, Eigen::Vector2d(x, y), pt_curr);
            
            // update the depth map
            updateDepthFilter(Eigen::Vector2d(x, y), pt_curr, T_c_r, depth, depth_cov);
        }
    }
    return true;
}

bool epipolarSearch(const cv::Mat& ref, const cv::Mat& curr, const Sophus::SE3& T_c_r, const Eigen::Vector2d& pt_ref, const double& depth_mu, const double& depth_cov, Eigen::Vector2d& pt_curr)
{
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Eigen::Vector3d p_ref = f_ref*depth_mu;
    
    Eigen::Vector2d px_mean_curr = cam2px(T_c_r * p_ref);           // 按深度均值投影的像素
    double d_min = depth_mu - 3*depth_cov, d_max = depth_mu + 3*depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Eigen::Vector2d px_min_curr = cam2px(T_c_r * (f_ref * d_min));  // 按最小深度投影的像素
    Eigen::Vector2d px_max_curr = cam2px(T_c_r * (f_ref * d_max));  // 按最大深度投影的像素
    
    Eigen::Vector2d epipolar_line = px_max_curr - px_min_curr;      // 极线（线段形式）
    Eigen::Vector2d epipolar_direction = epipolar_line;             // 极线方向 
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();                // 极线线段的半长度
    if (half_length > 100) half_length = 100;                       // 我们不希望搜索太多东西 
    
    // 取消此句注释以显示极线（线段）
//     showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);
    
    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;
    Eigen::Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7){      // l += sqrt(2)
        Eigen::Vector2d px_curr = px_mean_curr + l * epipolar_direction;    // 待匹配点
        if (!inside(px_curr))
            continue;
        // 计算待匹配点与参考帧的 NCC
        double ncc = computeNCC(ref, curr, pt_ref, px_curr);
    }
        
}























