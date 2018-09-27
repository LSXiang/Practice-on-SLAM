#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "DBoW3/DBoW3.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: feature_training <path to data>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // read the image
    std::cout << "reading images..." << std::endl;
    
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; i++) {
        std::string path_name = (std::string)(argv[1]) + "/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path_name));
    }
    
    // detect ORB features 
    std::cout << "detecting ORB features..." << std::endl;
    
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& image : images) {
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), key_points, descriptor);
        descriptors.push_back(descriptor);
    }
    
    // create vocabulary
    std::cout << "creating vocabulary ..." << std::endl;
    
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    
    std::cout << "vocabulary info: " << vocab << std::endl;
    
    vocab.save("vocabulary.yml.gz");
    
    std::cout << "done." << std::endl;
    
    return 0u;
}
