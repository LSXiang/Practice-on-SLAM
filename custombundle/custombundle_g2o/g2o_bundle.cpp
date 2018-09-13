#include <iostream>
#include <stdint.h>

#include <Eigen/StdVector>
#include <Eigen/Core>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "g2o/types/sba/types_six_dof_expmap.h"

#include "BundleParams.h"
#include "BALProblem.h"
#include "bal_class_g2o.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BalBlockSolver;

/**
 * set up the vertexs and edges for the bundle adjustment
 */
void buildProblem(const BALProblem* bal_problem, g2o::SparseOptimizer* optimizer, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();
    
    /* Set camera vertexs with initial value in the dataset */
    const double* raw_cameras = bal_problem->cameras();
    for (int i = 0; i < num_cameras; ++i) {
        ConstVectorRef temVecCamera(raw_cameras + camera_block_size * i, camera_block_size);
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        pCamera->setEstimate(temVecCamera);     // initial value for the camera i
        pCamera->setId(i);                      // set id for each camera vertex 
        
        /* remeber to add vertex into optimizer */
        optimizer->addVertex(pCamera);
    }
    
    /* Set point vertex with initial value in the dataset. */
    const double* raw_points = bal_problem->points();
    for (int i = 0; i < num_points; ++i) {
        ConstVectorRef temVecPoint(raw_points + point_block_size * i, point_block_size);
        VertexPointBAL* pPoint = new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);       // initial value for the point i
        pPoint->setId(num_cameras + i);         // each vertex should have an unique id, no matter it is a camera vertex, or a point vertex
        
        // remeber to add vertex into optimizer.
        pPoint->setMarginalized(true);
        optimizer->addVertex(pPoint);
    }
    
    /* Set edges for graph */
    const int num_observations = bal_problem->num_observations();
    const double* observations = bal_problem->observations();
    for (int i = 0; i < num_observations; ++i) {
        EdgeObservationBAL* bal_edge = new EdgeObservationBAL();
        
        const int camera_id = bal_problem->camera_index()[i];               // get id for the camera
        const int point_id = bal_problem->point_index()[i] + num_cameras;   // get id for the point
        
        if (params.robustify) {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.f);
            bal_edge->setRobustKernel(rk);
        }
        
        /* set the vertex by the ids for an edge observation */
        bal_edge->setVertex(0, dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        bal_edge->setVertex(1, dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
        bal_edge->setInformation(Eigen::Matrix2d::Identity());
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i +0], observations[2*i +1]));
        
        optimizer->addEdge(bal_edge);
    }
}

void writeToBALProblem(BALProblem* bal_problem, g2o::SparseOptimizer* optimizer)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();
    
    double* raw_cameras = bal_problem->mutable_cameras();
    for (int i = 0; i < num_cameras; ++i) {
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd newCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i * point_block_size, newCameraVec.data(), sizeof(double) * camera_block_size);
    }
    
    double* raw_points = bal_problem->mutable_points();
    for(int i = 0; i < num_points; ++i) {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(i + num_cameras));
        Eigen:: Vector3d newPointVec = pPoint->estimate();
        memcpy(raw_points + i * point_block_size, newPointVec.data(), sizeof(double) * point_block_size);
    }
}

int main(int argc, char* argv[])
{
    return 0;
}




























