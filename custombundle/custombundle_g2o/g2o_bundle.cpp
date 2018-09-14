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
        memcpy(raw_cameras + i * camera_block_size, newCameraVec.data(), sizeof(double) * camera_block_size);
    }
    
    double* raw_points = bal_problem->mutable_points();
    for(int i = 0; i < num_points; ++i) {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(i + num_cameras));
        Eigen:: Vector3d newPointVec = pPoint->estimate();
        memcpy(raw_points + i * point_block_size, newPointVec.data(), sizeof(double) * point_block_size);
    }
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, g2o::SparseOptimizer* optimizer)
{
    BalBlockSolver* solver_ptr;
    
    g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = nullptr;
    
    if (params.linear_solver == "dense_schur") {
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    } else if (params.linear_solver == "sparse_schur") {
        linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>*>(linearSolver)->setBlockOrdering(true);  // AMD ordering, only needed for sparse cholesky solver
    }
    
    solver_ptr = new BalBlockSolver(std::unique_ptr<g2o::LinearSolver<BalBlockSolver::PoseMatrixType>>(linearSolver));
    
    g2o::OptimizationAlgorithmWithHessian* solver;
    
    if (params.trust_region_strategy == "levenberg_marquardt") {
        solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<BalBlockSolver>(solver_ptr));
    } else if (params.trust_region_strategy == "dogleg") {
        solver = new g2o::OptimizationAlgorithmDogleg(std::unique_ptr<BalBlockSolver>(solver_ptr));
    } else {
        std::cout << "Please check your trust_region_strategy parameter again." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    optimizer->setAlgorithm(solver);
}

void solveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);
    
    /* show some information heer */
    std::cout << "BAL problem file loaded..." << std::endl;
    std::cout << "BAL problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points." << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations." << std::endl;
    
    /* store the initial 3d cloud points camera pose */
    if (!params.initial_ply.empty()) {
        bal_problem.WriteToPLYFile(params.initial_ply);
    }
    
    std::cout << "beginning problem..." << std::endl;
    
    /* add some noise for the initial value */
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma, params.point_sigma);
    
    std::cout << "Normalization complete..." << std::endl;
    
    g2o::SparseOptimizer optimizer;
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);
    buildProblem(&bal_problem, &optimizer, params);
    
    std::cout << "begin optimization ..." << std::endl;
    
    /* preform the optimization */
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);
    
    std::cout << "optimization complete." << std::endl;
    writeToBALProblem(&bal_problem, &optimizer);
    
    if (!params.final_ply.empty())  {
        bal_problem.WriteToPLYFile(params.final_ply);
    }
}

int main(int argc, char* argv[])
{
    BundleParams params(argc, argv);    // set the parameters here.
    
    if (params.input.empty()) {
        std::cout << "Usage: bundle_adjuster - input <path for dataset>";
        return 1;
    }
    
    solveProblem(params.input.c_str(), params);
    
    return 0;
}




























