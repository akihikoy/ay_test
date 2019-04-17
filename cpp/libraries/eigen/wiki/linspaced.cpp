#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  print(Vector4d::LinSpaced(4,-2.0,2.0));
  print(VectorXd::LinSpaced(5,-2.0,2.0));
  print(VectorXd::LinSpaced(3,1.0,-1.0));

  return 0;
}
