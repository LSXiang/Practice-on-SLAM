#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "projection.h"

class VectexCameraBAL : public g2o::BaseVertex<9, Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VectexCameraBAL() {}
    
    virtual bool read(std::istream & is)
    {
        return false;
    }
    
    virtual bool write(std::ostream & os) const 
    {
        return false;
    }
    
    virtual void setToOriginImpl() {}
    
    virtual void oplusImpl(const double *update)
    {
        Eigen::VectorXd::ConstMapType v(update, VectexCameraBAL::Dimension);
        _estimate += v;
    }
};
