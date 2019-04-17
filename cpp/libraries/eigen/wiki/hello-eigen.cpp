#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(2,3);  // サイズ 2x3 の double 型行列
  VectorXd v(3);  // サイズ 3x1 の double 型(列)ベクトル
  m(0,0) = 3.0;
  m(1,0) = 2.5;
  m(0,1) = -1.0;
  m(1,1) = m(1,0) + m(0,1);
  m(0,2) = 1.0e+10;
  m(1,2) = 2.0e-10;
  v(0)= 0.0;
  v(1)= -1.0;
  v(2)= 0.0;
  print(m);
  print(v);
  print(m*v);
  return 0;
}
