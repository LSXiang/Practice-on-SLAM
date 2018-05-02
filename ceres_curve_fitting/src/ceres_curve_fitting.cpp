/**
 * @file    ceres_curve_fitting.cpp
 * @version v1.0.0
 * @date    Sep,21 2017
 * @author  Jacob.lin
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

/* Calibration module of cost function */
struct curve_fitting_cost {
    curve_fitting_cost(double x, double y) : _x(x), _y(y) {}
    
    template <typename T>
    bool operator() (
        const T *const abc,     /* parameters for calibration module */
        T *residual ) const {   /* residual */
            
        // y-exp(ax^2+bx+c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
    const double _x, _y;
};

int main(int argc, char **argv)
{
    double a = 1.0, b = 2.0, c = 1.0;   /* actual parameters */
    int N = 100;                        /* data mumbers */
    double w_sigma = 1.0;               /* noise of sigma value */
    cv::RNG rng;                        /* gennerating random mumber from OpenCV */
    double abc[3] = {0, 0, 0};
    
    vector<double> x_data, y_data;
    
    cout << "generating data: " << endl;
    for (int i = 0; i < N; i ++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(ceres::exp(a*x*x + b*x + c) + rng.gaussian(w_sigma));
        cout << "x: "<< x_data[i] << "    y: " << y_data[i] << endl;
    }
    
    /* Constructing Least Squares Problem */ 
    ceres::Problem problem;
    for (int i = 0; i < N; i ++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<curve_fitting_cost, 1, 3>(new curve_fitting_cost(x_data[i], y_data[i])), nullptr, abc);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;     /* Optimization information */
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a:abc) {
        cout << a << " ";
    }
    cout << endl;
    
    return 0;
}



















