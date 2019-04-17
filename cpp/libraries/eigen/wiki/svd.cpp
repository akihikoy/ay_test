#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(2,3);
  m<< 3,2,1,
      -1,2,1;
  JacobiSVD<MatrixXd> svd(m, ComputeThinU | ComputeThinV);
  MatrixXd U,S,V;
  print(m);
  print(S=svd.singularValues());
  print(U=svd.matrixU());
  print(V=svd.matrixV());
  print(U * S.asDiagonal() * V.transpose());

  return 0;
}
