#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(2,3);
  VectorXd v(5);
  m<< 1,2,3,
      4,5,6;
  v<< -1,0,1,2,3;
  print(m);
  print(v);
  print(m.size());
  print(m.rows());
  print(m.cols());
  print(v.size());
  print(v.rows());
  print(v.cols());
  print(m.transpose());
  print(v.transpose());
  print(v.norm());

  return 0;
}
