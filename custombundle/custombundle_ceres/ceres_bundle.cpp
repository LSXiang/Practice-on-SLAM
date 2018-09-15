#include <iostream>
#include <fstream>
#include "ceres/ceres.h"

#include "SnavelyReprojectionError.h"
#include "BALProblem.h"
#include "BundleParams.h"

void setLinearSolver(ceres::Solver::Options* options, const BundleParams& params)
{
    CHECK(ceres::StringToLinearSolverType(params.linear_solver, &options->linear_solver_type));
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library, &options->sparse_linear_algebra_library_type));
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library, &options->dense_linear_algebra_library_type));
    options->num_linear_solver_threads = params.num_threads;
}

void setOrdering(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int point_block_size = bal_problem->point_block_size();
    double* points = bal_problem->mutable_points();
    
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    double* cameras = bal_problem->mutable_cameras();
    
    if (params.ordering == "automatic") {
        return;
    }
    
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
    
    /* The points come before the cameras */
    for (int i = 0; i < num_points; ++i) {
        ordering->AddElementToGroup(points + point_block_size * i, 0);
    }
    
    for (int i = 0; i < num_cameras; ++i) {
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
    }
    
    options->linear_solver_ordering.reset(ordering);
}

void setMinimizerOptions(ceres::Solver::Options* options, const BundleParams& params)
{
    options->max_num_iterations = params.num_iterations;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = params.num_threads;
    
    CHECK(ceres::StringToTrustRegionStrategyType(params.trust_region_strategy, &options->trust_region_strategy_type));
}

void setSolverOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, ceres::Solver::Options* options)
{
    setMinimizerOptions(options, params);
    setLinearSolver(options, params);
    setOrdering(bal_problem, options, params);
}

void buildProblem(BALProblem* bal_problem, ceres::Problem* problem, const BundleParams& params)
{
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    const double* points = bal_problem->points();
    const double* cameras = bal_problem->cameras();
    
    /**
     * Observations is 2*num_observations long array observations [u_1, u_2, ... u_n],
     * where each u_i is two dimensional, the x and y position of the observaton.
     */
    const double* observations = bal_problem->observations();
    
    for (int i = 0; i < bal_problem->num_observations(); i++) {
        ceres::CostFunction* cost_function = nullptr;
        
        /**
         * Each Residuals block takes a points and a camera as input and
         * outputs a 2 dimensional residual
         */
        cost_function = SnavelyReprojectionError::create(observations[2*i], observations[2*i + 1]);
        
        /* if enable use Huber's loss function */
        ceres::LossFunction* loss_function = params.robustify ? new ceres::HuberLoss(1.f) : nullptr;
        
        /**
         * Each observation correspondes to a pair of a camera and a point which are identified by
         * camera_index()[i] and point_index()[i] respectively.
         */
        double* camera = (double*)(cameras + camera_block_size * bal_problem->camera_index()[i]);
        double* point = (double*)(points + point_block_size * bal_problem->point_index()[i]);
        
        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }
}

void solveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);
    
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
    
    ceres::Problem problem;
    buildProblem(&bal_problem, &problem, params);
    
    std::cout << "The proble is successfully build ..." << std::endl;
    
    ceres::Solver::Options options;
    setSolverOptionsFromFlags(&bal_problem, params, &options);
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    
    ceres::Solver::Summary summary;
    
    std::cout << "begin optimization ..." << std::endl;
    
    ceres::Solve(options, &problem, &summary);
    
    std::cout << "optimization complete." << std::endl;
    
    std::cout << summary.FullReport() << std::endl;
    
    if (!params.final_ply.empty())  {
        bal_problem.WriteToPLYFile(params.final_ply);
    }
}

int main(int argc, char** argv)
{
    BundleParams params(argc, argv);    // set the parameter here.
    
    google::InitGoogleLogging(argv[0]);
    std::cout << params.input << std::endl;
    if (params.input.empty()) {
        std::cout << "Usage: customBundle_Ceres -input <dataset>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    solveProblem(params.input.c_str(), params);
    
    return 0;
}

















