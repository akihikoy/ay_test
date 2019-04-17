#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  print(MatrixXd::Identity(3,3));
  print(MatrixXd::Identity(2,3));
  print(MatrixXd::Zero(2,3));
  print(MatrixXd::Ones(2,3));
  print(MatrixXd::Constant(2,3,2.5));
  print(MatrixXd::Random(2,3));

  print(Vector3d::UnitX());
  print(Vector3d::UnitY());
  print(Vector3d::UnitZ());
  print(VectorXd::Unit(4,2));
  print(VectorXd::Zero(2));
  print(VectorXd::Ones(2));
  print(VectorXd::Constant(2,3.14));
  print(VectorXd::Random(2));

  return 0;
}
