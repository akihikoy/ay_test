#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m1(3,2);
  print(m1);
  MatrixXd m2;
  print(m2);
  print(m2.cols());
  print(m2.rows());
  m2.resize(3,2);
  // m2.resize(9);
  print(m2);
  print(m2.cols());
  print(m2.rows());

  VectorXd v;
  print(v);
  print(v.size());
  print(v.cols());
  print(v.rows());
  v.resize(3);
  // v.resize(3,2);
  print(v);
  print(v.size());
  print(v.cols());
  print(v.rows());
  return 0;
}
