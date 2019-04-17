#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(3,3);
  m<< 5,2,1,
      2,8,1,
      1,1,5;
  LLT<MatrixXd> chol(m);
  MatrixXd L,U;
  print(m);
  print(L= chol.matrixL());
  print(L * L.transpose());

  return 0;
}
