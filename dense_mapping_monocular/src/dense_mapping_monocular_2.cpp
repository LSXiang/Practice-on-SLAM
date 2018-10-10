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

/* The version 2 uses inverse depth and adds affine transformation with respect to the version 1 */

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
    std::vector<Sophus::SE3>& poses
);

bool update(
    const cv::Mat& ref, 
    const cv::Mat& curr, 
    const Sophus::SE3& T_c_r, 
    cv::Mat& depth, 
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
    uchar* d = & image.data[int(pt.y()) * image.step + int(pt.x())];
    
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
    if (argc != 2) {
        std::cout << "Usage: dense_mapping_monocular <path_to_test_dataset>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // read the data form the dataset
    std::vector<std::string> color_image_files;
    std::vector<Sophus::SE3> poses_T_w_c;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_T_w_c);
    if (ret == false) {
        std::cout << "Reading files failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "read total " << color_image_files.size() << " files." << std::endl;
    
    // first image
    cv::Mat ref = cv::imread(color_image_files[0], 0);      // garg-scale image
    Sophus::SE3 pose_ref_Twc = poses_T_w_c[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    cv::Mat depth(height, width, CV_64F, init_depth);
    cv::Mat depth_cov(height, width, CV_64F, init_cov2);
    
    for (int index = 1; index < color_image_files.size(); index ++) {
        std::cout << "*** loop " << index << " ***" << std::endl;
        cv::Mat curr = cv::imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        Sophus::SE3 pose_curr_Twc = poses_T_w_c[index];
        Sophus::SE3 pose_curr_ref = pose_curr_Twc.inverse() * pose_ref_Twc;
        update(ref, curr, pose_curr_ref, depth, depth_cov);
        plotDepth(depth);
        cv::imshow("image", curr);
        cv::waitKey(1);
    }
    
    std::cout << "estimation returens, saving depth map ..." << std::endl;
    cv::imwrite("depth.png", depth);
    std::cout << "done." << std::endl;
    
    return 0u;
}

bool readDatasetFiles(const std::string& path, std::vector<std::string>& color_image_files, std::vector<Sophus::SE3>& poses)
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
        poses.push_back(
            Sophus::SE3(Sophus::Quaterniond(data[6], data[3], data[4], data[5]),
                        Sophus::Vector3d(data[0], data[1], data[2]))
        );
        
        if (!fin.good()) break;
    }
    
    return true;
}

/* update the global depth map */
bool update(const cv::Mat& ref, const cv::Mat& curr, const Sophus::SE3& T_c_r, cv::Mat& depth, cv::Mat& depth_cov)
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
    
    // if the epipolar line length <= 1.5 pixel, we regard that the mean point is the matching point
    if (half_length <= 1.5) {
        pt_curr = px_mean_curr;
        if (!inside(pt_curr))
            return false;
        return true;
    } 
    
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
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)       // 只相信 NCC 很高的匹配
        return false;
    
    pt_curr = best_px_curr;
    
    return true;
}

double computeNCC(const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& pt_ref, const Eigen::Vector2d& pt_curr)
{
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    std::vector<double> values_ref, values_curr;    // 参考帧和当前帧的均值
    for (int x = -ncc_window_size; x <= ncc_window_size; x ++) {
        for (int y = -ncc_window_size; y <= ncc_window_size; y ++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref.y()))[int(x + pt_ref.x())]) / 255.0;
            mean_ref += value_ref;
            
//             double value_curr = double(curr.ptr<uchar>(int(y + pt_curr.y()))[int(x + pt_curr.x())]) / 255.0;
            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
            mean_curr += value_curr;
            
            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
    }
    
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;
    
    // compute Zero mean NCC 
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10);   // 防止分母出现零
}

bool updateDepthFilter(const Eigen::Vector2d& pt_ref, const Eigen::Vector2d& pt_curr, const Sophus::SE3& T_c_r, cv::Mat& depth, cv::Mat& depth_cov)
{
    // 用三角化计算深度
    Sophus::SE3 T_r_c = T_c_r.inverse();
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Eigen::Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();
    
    /**
     * function:
     * d_ref * f_ref = d_cur * (R_r_c * f_cur) + t_r_c
     * => [f_ref^T f_ref, -f_ref^T f_cur] [d_ref] = [f_ref^T t]
     *    [f_cur^T f_ref, -f_cur^T f_cur] [d_cur] = [f_cur^T t]
     * 二阶方程用克莱默法则求解并解之
     */
    Eigen::Vector3d t = T_r_c.translation();
    Eigen::Vector3d f2 = T_r_c.rotation_matrix() * f_curr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
    
    double A[4];
    A[0] = f_ref.dot(f_ref);
    A[2] = f_ref.dot(f2);
    A[1] = -A[2];
    A[3] = -f2.dot(f2);
    
    double d = A[0]*A[3] - A[1]*A[2];
    Eigen::Vector2d lambda_vec = Eigen::Vector2d(
        A[3]*b(0, 0) - A[1]*b(1, 0),
        A[0]*b(1, 0) - A[2]*b(0, 0)) / d;
    
    Eigen::Vector3d xm = lambda_vec(0, 0) * f_ref;
    Eigen::Vector3d xn = lambda_vec(1, 0) * f2 + t;
    Eigen::Vector3d d_esti = (xm + xn) / 2.0;       // 三角化算得的深度向量
    double depth_estimation = d_esti.norm();        // 深度值
    
    // 计算不确定性（以一个像素为误差）
    Eigen::Vector3d p = f_ref*depth_estimation;
    Eigen::Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(a.dot(-t) / a_norm / t_norm);
    double beta_prime = beta + atan(1.0 / fx / 2.0) * 2.0;
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;
    
    
//     double tau_inverse = 0.5 * (1.0/std::max(0.0000001, depth_estimation-d_cov) - 1.0/(depth_estimation+d_cov));
//     double d_cov2 = (tau_inverse * tau_inverse);
    
    // 高斯融合
    double mu = 1.f / depth.ptr<double>(int(pt_ref.y()))[int(pt_ref.x())];
    double sigma2 =  depth_cov.ptr<double>(int(pt_ref.y()))[int(pt_ref.x())];
    
    double mu_fuse = (d_cov2*mu + sigma2*(1.f / depth_estimation)) / (d_cov2 + sigma2);
    double sigma2_fuse = (d_cov2*sigma2) / (d_cov2 + sigma2);
    
    depth.ptr<double>(int(pt_ref.y()))[int(pt_ref.x())] = 1.f / mu_fuse;
    depth_cov.ptr<double>(int(pt_ref.y()))[int(pt_ref.x())] =  sigma2_fuse;
    
    return true;
}

void plotDepth(const cv::Mat& depth)
{
    cv::imshow("depth", depth * 0.4);
    cv::waitKey(1);
}

void showEpipolarMatch(const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& px_ref, const Eigen::Vector2d& px_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);
    
    cv::circle(ref_show, cv::Point2f(px_ref.x(), px_ref.y()), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr.x(), px_curr.y()), 5, cv::Scalar(0, 0, 250), 2);
    
    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void showEpipolarLine(const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& px_ref, const Eigen::Vector2d& px_mincurr, const Eigen::Vector2d& px_maxcurr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);
    
    cv::circle(ref_show, cv::Point2f(px_ref.x(), px_ref.y()), 5, cv::Scalar(0, 250, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_mincurr.x(), px_mincurr.y()), 5, cv::Scalar(0, 250, 2), 2);
    cv::circle(curr_show, cv::Point2f(px_maxcurr.x(), px_maxcurr.y()), 5, cv::Scalar(0, 250, 2), 2);
    cv::line(curr_show, cv::Point2f(px_mincurr.x(), px_mincurr.y()), cv::Point2f(px_maxcurr.x(), px_maxcurr.y()), cv::Scalar(0, 250, 0), 1);
    
    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}







































