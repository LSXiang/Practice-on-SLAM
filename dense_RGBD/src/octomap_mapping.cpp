#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>

#include <boost/format.hpp>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#define OCTOMAP_WITH_COLOR      1

int main(int argc, char** argv)
{
    std::vector<cv::Mat> color_images, depth_images;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    
    std::ifstream fin("../../data/pose.txt");
    if (!fin) {
        std::cerr << "please run the program with the directory -'../../data/' and file 'pose.txt'.";
        exit(EXIT_FAILURE);
    }
    
    for (int i=0; i < 5; i ++) {
        boost::format fmt("../../data/%s/%d.%s");
        color_images.push_back(cv::imread((fmt%"color"%(i+1)%"png").str()));
        depth_images.push_back(cv::imread((fmt%"depth"%(i+1)%"pgm").str(), -1));
        
        double data[7] = {0};
        for (int i = 0; i < 7; i ++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }
    
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    
    std::cout << "The image is being converted to a OctoMap ..." << std::endl;
    
    // octomap tree
#if OCTOMAP_WITH_COLOR
    struct RGB_Color {
        uchar r;
        uchar g;
        uchar b;
    };
    
    octomap::ColorOcTree tree(0.05);
#else
    octomap::OcTree tree(0.05);     // Resolution
#endif
    
    for (int i = 0; i < 5; i++) {
        std::cout << "Image "<< i+1 << " is being converted ... " << std::endl;
        
        cv::Mat color = color_images[i];
        cv::Mat depth = depth_images[i];
        Eigen::Isometry3d T = poses[i];
        
        octomap::Pointcloud cloud;  // the point cloud in octomap
#if OCTOMAP_WITH_COLOR
        std::vector<RGB_Color> rgb_colors;
#endif
        
        for (int v = 0; v < color.rows; v++) {
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u];   // depth value
                if (d == 0) continue;       /* last measure, value is 0 */
                if (d >= 7000) continue;    /* the depth value too large */
                
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                Eigen::Vector3d point_world = T*point;
                
                cloud.push_back(point_world[0], point_world[1], point_world[2]);
#if OCTOMAP_WITH_COLOR
                RGB_Color rgb_color;
                rgb_color.b = color.ptr<uchar>(v)[3*u];
                rgb_color.g = color.ptr<uchar>(v)[3*u + 1];
                rgb_color.r = color.ptr<uchar>(v)[3*u + 2];
                rgb_colors.push_back(rgb_color);
#endif
            }
        }
        
        tree.insertPointCloud(cloud, octomap::point3d(T(0,3), T(1,3), T(2,3)));
        
#if OCTOMAP_WITH_COLOR
        int j = 0;
//         for (octomap::point3d_collection::iterator p = cloud.begin(); p < cloud.end(); p ++) {
        for (auto p = cloud.begin(); p < cloud.end(); p++) {
            auto rgb = rgb_colors.data()[j];
            tree.integrateNodeColor((*p).x(), (*p).y(), (*p).z(), rgb.r, rgb.g, rgb.b);
            j ++;
        }
#endif
    }
    
    tree.updateInnerOccupancy();
    std::cout << "saving octomap..." << std::endl;

#if OCTOMAP_WITH_COLOR
    tree.write("octomap.ot");
#else
    tree.writeBinary("octomap.bt");
#endif

    return 0;
}





























