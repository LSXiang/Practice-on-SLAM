#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

int main (int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: pose_graph_gtsam sphere.g2o" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cout << "file " << argv[1] << " does not exist." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    gtsam::NonlinearFactorGraph::shared_ptr factor_graph_ptr(new gtsam::NonlinearFactorGraph);  // gtsam of factor graph
    gtsam::Values::shared_ptr initial_ptr(new gtsam::Values);                                   // values of initialized
    
    /**
     * get vertices & edges information form the sphere.g2o files.
     */
    int cntVertex = 0, cntEdge = 0;
    std::cout << "reading from g2o file." << std::endl;
    
    while (!fin.eof()) {
        std::string name;
        fin >> name;
        
        if (name == "VERTEX_SE3:QUAT") {
            // vertex 
            gtsam::Key id;
            fin >> id;
            
            double data[7];
            for (int i = 0; i < 7; i++) {
                fin >> data[i];
            }
            gtsam::Rot3 R = gtsam::Rot3::quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            initial_ptr->insert(id, gtsam::Pose3(R, t));        // add initial values
            cntVertex ++;

        } else if (name == "EDGE_SE3:QUAT") {
            gtsam::Matrix m = gtsam::I_6x6;             // information matrix
            gtsam::Key idx1, idx2;
            fin >>  idx1 >> idx2;
            double data[7];
            for (int i = 0; i < 7; i++) 
                fin >> data[i];
            
            gtsam::Rot3 R = gtsam::Rot3::quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            for (int i = 0; i < 6; i ++ ) {
                for (int j = i; j < 6; j ++) {
                    double mij;
                    fin >> mij;
                    m(i, j) = mij;
                    m(j, i) = mij;
                }
            }
            
            // information matrix
            gtsam::Matrix mgsam_information = gtsam::I_6x6;
            mgsam_information.block<3, 3>(0, 0) = m.block<3, 3>(3, 3);  // cov rotation
            mgsam_information.block<3, 3>(3, 3) = m.block<3, 3>(0, 0);  // cov translation
            mgsam_information.block<3, 3>(0, 3) = m.block<3, 3>(0, 3);  // off diagonal
            mgsam_information.block<3, 3>(3, 0) = m.block<3, 3>(3, 0);  // off diagonal
            
            gtsam::SharedNoiseModel model = gtsam::noiseModel::Gaussian::Information(mgsam_information);    // Gaussian noise model
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(idx1, idx2, gtsam::Pose3(R, t), model)); // add a factor
            factor_graph_ptr->push_back(factor);
            cntEdge ++;
        }
        
        if (!fin.good()) break;
    }
    
    std::cout << "read total " << cntVertex << " vertices, " << " cntEdge." << std::endl;
    
    /* 固定第一个顶点，在gtsam中相当于添加一个先验因子 */
    gtsam::NonlinearFactorGraph graph_with_prior = *factor_graph_ptr;
    gtsam::noiseModel::Diagonal::shared_ptr prior_model = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished()
    );
    
    gtsam::Key first_key = 0;
    
//     for(const gtsam::Values::ConstKeyValuePair& key_value : *initial_ptr) {
    for (auto key_value : *initial_ptr) {
        std::cout << "Adding prior to g2o file" << std::endl;
        graph_with_prior.add(gtsam::PriorFactor<gtsam::Pose3>(
            key_value.key, key_value.value.cast<gtsam::Pose3>(), prior_model)
        );
        break;
    }
    
    // 开始因子图优化，配置优化选项
    std::cout << "optimizing the factor graph." << std::endl;
    
    // use LM optimization
    gtsam::LevenbergMarquardtParams params_lm;
    params_lm.setVerbosity("ERROR");
    params_lm.setMaxIterations(20);
    params_lm.setLinearSolverType("MULTIFRONTAL_QR");
    gtsam::LevenbergMarquardtOptimizer optimizer_LM(graph_with_prior, *initial_ptr, params_lm);
    
    // try use GN
//     gtsam::GaussNewtonParams params_gn;
//     params_gn.setVerbosity("ERROR");
//     params_gn.setMaxIterations(20);
//     params_gn.setLinearSolverType("MULTIFRONTAL_QR");
//     gtsam::GaussNewtonOptimizer optimizer_GN(graph_with_prior, *initial_ptr, params_gn);
    
    gtsam::Values result = optimizer_LM.optimize();
    
    std::cout << "optimization complete." << std::endl;
    
    std::cout << "initial error: " << factor_graph_ptr->error(*initial_ptr) << std::endl;
    std::cout << "final error: " << factor_graph_ptr->error(result) << std::endl;
    
    std::cout << "done.\r\nwrite to g2o ..." << std::endl;
    
    std::ofstream fout("result_gtsam.g2o");
    // vertex
    for (auto key_value : result) {
//     for (const gtsam::Values::ConstKeyValuePair& key_value : result) {
        gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
        gtsam::Point3 t = pose.translation();
        gtsam::Quaternion q = pose.rotation().toQuaternion();
        fout << "VERTEX_SE3:QUAT " << key_value.key << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    // edge 
    for (auto factor : *factor_graph_ptr) {
//     for (gtsam::NonlinearFactor::shared_ptr factor : *factor_graph_ptr) {
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr f = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
//         gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr f = std::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);    // this program is error !!!!!!
        if (f) {
            gtsam::SharedNoiseModel model = f->noiseModel();
            gtsam::noiseModel::Gaussian::shared_ptr gaussianModel = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(model);
            if (gaussianModel) {
                gtsam::Matrix info = gaussianModel->R().transpose() * gaussianModel->R();
                gtsam::Pose3 pose = f->measured();
                gtsam::Point3 t = pose.translation();
                gtsam::Quaternion q = pose.rotation().toQuaternion();
                
                fout << "EDGE_SE3:QUAT " << f->key1() << " " << f->key2() << " "
                     << t.x() << " " << t.y() << " " << t.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";
                
                gtsam::Matrix infoG2O = gtsam::I_6x6;
                infoG2O.block(0, 0, 3, 3) = info.block(3, 3, 3, 3);     // cov translation
                infoG2O.block(3, 3, 3, 3) = info.block(3, 3, 3, 3);     // cov rotation
                infoG2O.block(0, 3, 3, 3) = info.block(0, 3, 3, 3);     // off diagonal
                infoG2O.block(3, 0, 3, 3) = info.block(3, 0, 3, 3);     // off diagonal
                
                for (int i = 0; i < 6; i++) {
                    for (int j = i; j < 6; j++) {
                        fout << infoG2O(i, j) << " ";
                    }
                }
                
                fout << std::endl;
            }
        }
    }
    
    fout.close();
    std::cout << "done." << std::endl;
    
    return 0;
}









