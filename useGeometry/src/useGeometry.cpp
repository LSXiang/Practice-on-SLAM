/**
 * @file    useGeometry.cpp
 * @date    2017/9/8
 * @version V1.0
 * @author  Jacob.lin
 */

#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

int main(int argc, char **argv)
{
    argc = (int)argc;
    argv = (char **)argv;
    
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d(0, 0, 1));
    
    cout.precision(3);
    cout << "rotation_matrix = \r\n" << rotation_vector.matrix() << endl;
    
    rotation_matrix = rotation_vector.toRotationMatrix();
    
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotation = rotation_vector * v;
    cout << "(1, 0, 0) after rotation = " << v_rotation.transpose() << endl;
    
    v_rotation = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation = " << v_rotation.transpose() << endl;
    
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);    /* 2,1,0 is ZYX axis, yaw, pitch, roll */
    cout << "yaw pitch roll = " << euler_angles.transpose();
    
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1, 3, 4));
    cout << "Transform matrix = \r\n" << T.matrix() << endl;
    
    Eigen::Vector3d v_transformed = T * v;
    cout << "v tranformed = " << v_transformed.transpose() << endl;
    
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    cout << "quaternion = \r\n" << q.coeffs() << endl;
    q = Eigen::Quaterniond(rotation_matrix);
    cout << "quaternion = \r\n" << q.coeffs() << endl;
    
    v_rotation = q * v; /* qvq^(-1) */
    cout << "(1, 0, 0) after rotation = " << v_rotation.transpose() << endl;
    
    return 0;
}


