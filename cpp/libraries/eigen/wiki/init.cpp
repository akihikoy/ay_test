#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  Vector2d v2(3.0,2.0);
  print(v2);
  Vector3d v3(-1.0,3.0,2.0);
  print(v3);
  Vector4d v4(-1.0,3.0,2.0,4.0);
  print(v4);
  double a[3]= {0.1,0.2,0.3};
  Vector3d v3a(a);
  print(v3a);
  return 0;
}
