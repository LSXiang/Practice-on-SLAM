/**
 * File:    imageBasics.cpp
 * Version: V1.0.0
 * Date:    Sep,14 2017
 * Author:  Jacob.lin
 */

#include<iostream>
#include<chrono>
using namespace std;

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);
    
    if (image.data == nullptr) {
        cerr << "The file " << argv[1] << " can't find." << endl;
        return 0;
    }
    
    
    cout << "The width of image : " << image.cols << ", the high of image : " << image.rows \
         << ", the channels of image : " << image.channels() << endl;
    cv::imshow("image", image);
    cv::waitKey(0);
    
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        cout << "Please input a color photo or grayscale photo." << endl;
        return 0;
    }
    
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; y++) {
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        
        for(size_t x = 0; x < image.cols; x++) {
            unsigned char *data_ptr = &row_ptr[x*image.channels()];
            for (int c = 0; c != image.channels(); c++) {
                unsigned char data = data_ptr[c];
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "The time it takes to traverse the image : " << time_used.count() << "sec" << endl;
    
    cv::Mat image_another = image;
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::imshow("image", image);
    cv::waitKey(0);
    
    
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);
    
    cv::destroyAllWindows();
    return 0;
}
