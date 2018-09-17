
#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>


/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化,
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 ***********************************************/

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cout << "File " << argv[1] << "does not exits." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;   // 6x6 BlockSolver
    g2o::LinearSolver<Block::PoseMatrixType>* linear_solver_ptr = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();    // linear solver
    
    Block* block_solver_ptr = new Block(std::unique_ptr<g2o::LinearSolver<Block::PoseMatrixType>>(linear_solver_ptr));      // block solver 
    
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(block_solver_ptr));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);     // 设置求解器
    
    int vertexCnt = 0, edgeCnt = 0;     // the number of vertex & edge
    while (!fin.eof()) {
        std::string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 vertex
            g2o::VertexSE3* v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt ++;
            if (index == 0)
                v->setFixed(true);
                    
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3 - SE3 dege
            g2o::EdgeSE3* edge = new g2o::EdgeSE3();
            int index1, index2;     // 关联的两个顶点
            fin >> index1 >> index2;
            edge->setId(edgeCnt ++);
            edge->setVertex(0, optimizer.vertices()[index1]);
            edge->setVertex(1, optimizer.vertices()[index2]);
            edge->read(fin);
            optimizer.addEdge(edge);
        }
        
        if (!fin.good()) break;
    }
    
    std::cout << "read total " << vertexCnt << " virtices, " << edgeCnt << " edges." << std::endl;
    
    std::cout << "prepare optimizing ..." << std::endl;
    
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    
    std::cout << "calling optimizing ..." << std::endl;
    
    optimizer.optimize(30);
    
    std::cout << "saving optimization results ..." << std::endl;
    optimizer.save("result_SE3.g2o");
    
    return 0;
}







