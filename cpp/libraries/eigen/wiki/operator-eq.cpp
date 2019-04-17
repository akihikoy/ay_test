#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  RowVector4d v1(-1.0,3.0,2.0,1.0);
  VectorXd v2;
  MatrixXd m1;
  Matrix2d m2;
  v2=v1;
  m1=v1;
  // m2=v1;  // これはエラー
  print(v1);
  print(v2);
  print(m1);
  return 0;
}
