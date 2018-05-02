/**
 * File:    useSophus.cpp
 * Version: V1.0.0
 * Date:    Sep,12 2017
 * Author:  Jacob.lin
 */

#include<iostream>
#include<cmath>
using namespace std;

#include<Eigen/Core>
#include<Eigen/Geometry>
using namespace Eigen;

#include<sophus/so3.h>
#include<sophus/se3.h>

int main(int argc, char **argv)
{
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Sophus::SO3 SO3_R(R);
    Sophus::SO3 SO3_v(0, 0, M_PI_2);
    Eigen::Quaterniond q(R);
    Sophus::SO3 SO3_q(q);
    
    cout << "R = \r\n" << R << endl;
    
    cout << "SO(3) from matrix:\r\n" << SO3_R.matrix() << endl;
    cout << "SO(3) from vector:\r\n" << SO3_v << endl;
    cout << "SO(3) from quaternion:\r\n" << SO3_q << endl;
    
    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 = \r\n" << so3.transpose() << endl;
    cout << "so3 hat = \r\n" << Sophus::SO3::hat(so3) << endl;
    cout << "so3 hat vee = \r\n" << Sophus::SO3::vee(Sophus::SO3::hat(so3)).transpose() << endl;
    
    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \r\n" << SO3_updated << endl;
    cout << "SO3 updated = \r\n" << Sophus::SO3::hat(SO3_updated.log()) << endl;
    
    Eigen::Vector3d t(1, 0, 0);
    Sophus::SE3 SE3_Rt(R, t);
    Sophus::SE3 SE3_qt(q, t);
    cout << "SE3 from R,t =\r\n" << SE3_Rt.matrix() << endl;
    cout << "SE3 from q,t =\r\n" << SE3_qt << endl;
    
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 =\r\n" << se3.transpose() << endl;
    cout << "se3 hat =\r\n" << Sophus::SE3::hat(se3) << endl;
    cout << "se3 hat vee =\r\n" << Sophus::SE3::vee(Sophus::SE3::hat(se3)).transpose() << endl;
    
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0) = 1e-4d;
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3) * SE3_Rt;
    cout << "SE3 update = \r\n" << SE3_updated.matrix() << endl;
    
    return 0;
}
