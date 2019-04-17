#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(3,3);
  m<< 3,2,1,
      2,2,1,
      1,1,1;
  SelfAdjointEigenSolver<MatrixXd> eig(m);
  MatrixXd D,V;
  print(m);
  print(D=eig.eigenvalues());
  print(V=eig.eigenvectors());
  print(V * D.asDiagonal() * V.transpose());
  print(V * D.asDiagonal() * V.inverse());

  return 0;
}
