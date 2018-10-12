#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <sophus/se3.h>
#include <sophus/so3.h>

typedef Eigen::Matrix<double, 6, 6> Matrix6d;

/**
 * Give an approximation( J_R^{-1} ) of the error
 */
Matrix6d JRInv(Sophus::SE3 error)
{
    Matrix6d Jacobian;
    Jacobian.block(0, 0, 3, 3) = Sophus::SO3::hat(error.so3().log());
    Jacobian.block(0, 3, 3, 3) = Sophus::SO3::hat(error.translation()); // * Sophus::SO3::hat(error.so3().log());
    Jacobian.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
    Jacobian.block(3, 3, 3, 3) = Sophus::SO3::hat(error.so3().log());
    
    Jacobian = Jacobian*0.5 + Matrix6d::Identity();
    
    return Jacobian;
}

/**
 * vertex of lie algebra
 */
typedef Eigen::Matrix<double, 6, 1> Vector6d;
class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    bool read(std::istream& is)
    {
        double data[7];
        for (int i = 0; i < 7; i ++) {
            is >> data[i];
        }
        
        setEstimate(Sophus::SE3(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Eigen::Vector3d(data[0], data[1], data[2])
        ));
        
        return true;
    }
    
    bool write(std::ostream &os) const
    {
        os << id() << " ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " <<q.coeffs()[3] << std::endl;
        
        return true;
    }
    
    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3();
    }
    
    /* update */
    virtual void oplusImpl(const double * update)
    {
        Sophus::SE3 up(
            Sophus::SO3(update[3], update[4], update[5]),
            Eigen::Vector3d(update[0], update[1], update[2])
        );
        _estimate = up * _estimate;
    }
};

class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3, VertexSE3LieAlgebra, VertexSE3LieAlgebra>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    bool read(std::istream& is)
    {
        double data[7];
        for (int i = 0; i < 7; i++) {
            is >> data[i];
        }
        
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(Sophus::SE3(q, Eigen::Vector3d(data[0], data[1], data[2])));
        
        for (int i = 0; i < information().rows() && is.good(); i ++) {
            for (int j = i; j < information().cols() && is.good(); j ++) {
                is >> information()(i, j);
                if (i != j) {
                    information()(j, i) = information()(i, j);
                }
            }
        }
        
        return true;
    }
    
    bool write(std::ostream& os) const
    {
        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*>(_vertices[1]);
        
        os << v1->id() << " " << v2->id() << " ";
        
        Sophus::SE3 m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << q.coeffs()[3] << " ";
        
        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++)
                os << information()(i, j) << " ";
        
        os << std::endl;
        
        return true;
    }
    
    // Compute Error
    virtual void computeError()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra*>(_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra*>(_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }
    
    // compute Jacobian
    virtual void linearizeOplus()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra*>(_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra*>(_vertices[1]))->estimate();
        Matrix6d J = JRInv(Sophus::SE3::exp(_error));
        
        // try J ~= I ?
        _jacobianOplusXi = - J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: pose_graph_g2o_lie_algebra sphere.g2o" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cout << "file " << argv[1] << " does not exist." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;
    Block::LinearSolverType* linear_solver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block* block_solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linear_solver));
    g2o::OptimizationAlgorithmLevenberg* optimization_algorithm_ptr = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(block_solver_ptr));
//     g2o::OptimizationAlgorithmGaussNewton* optimization_algorithm_ptr = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<Block>(block_solver_ptr));
//     g2o::OptimizationAlgorithmDogleg* optimization_algorithm_ptr = new g2o::OptimizationAlgorithmDogleg(std::unique_ptr<Block>(block_solver_ptr));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(optimization_algorithm_ptr);
    
    int vertexCnt = 0, edgeCnt = 0;     // the number of vertex & edge
    std::vector<VertexSE3LieAlgebra*> vertices;
    std::vector<EdgeSE3LieAlgebra*> edges;
    
    while (!fin.eof()) {
        std::string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // vertex
            VertexSE3LieAlgebra* v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt ++;
            vertices.push_back(v);
            if (index == 0)
                v->setFixed(true);
                    
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3 - SE3 dege
            EdgeSE3LieAlgebra* e = new EdgeSE3LieAlgebra();
            int idx1, idx2;
            fin >> idx1 >> idx2;
            e->setId(edgeCnt ++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        
        if (!fin.good()) break;
    }
    
    std::cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << std::endl;
    
    std::cout << "prepare optimizing ..." << std::endl;
    
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    
    std::cout << "calling optimizing ..." << std::endl;
    
    optimizer.optimize(30);
    
    std::cout << "saving optimization results ..." << std::endl;
    
    /* 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现伪装成 SE3 顶点和边，让 g2o_viewer 可以认出 */
    std::ofstream fout("result_lie.g2o");
    if (!fout.is_open()) {
        std::cout << "result_lie.g2o dose not exist." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (auto v : vertices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (auto e : edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    
    fout.close();
    
    return 0;
}



















