#ifndef __SNAVELYREPROJECTION_H__
#define __SNAVELYREPROJECTION_H__

#include <iostream>
#include "ceres/ceres.h"

#include "tools/rotation.h"
#include "projection.h"

class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x), observed_y(observation_y) {}
    
    template<typename T>
    bool operator() (const T* const camera, const T* const point, T* residuals) const 
    {
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        
        residuals[0] = predictions[0] - observed_x;
        residuals[1] = predictions[1] - observed_y;
        
        return true;
    }
    
    static ceres::CostFunction* create(const double observed_x, const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
    }
    
private:
    double observed_x;
    double observed_y;
};

#endif /* __SNAVELYREPROJECTION_H__ */
