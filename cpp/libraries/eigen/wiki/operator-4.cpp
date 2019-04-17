#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  Vector3d x(1.0,2.0,3.0);
  RowVector3d y(3.0,2.0,1.0);
  MatrixXd m(2,3);
  m<< 1.0,2.0,3.0,
      4.0,3.0,2.0;
  double a(2.0);
  print(x);
  print(y);
  print(m);
  print(a);
  print(m*x+Vector2d(1000.0,1000.0));
  print(y*x+1000.0);
  print((x*y)*a);
  print((y*(x*y))/a);
//   print(a*x);
//   print(a*y);
//   print(a*m);
//   print(x/a);
//   print(y/a);
//   print(m/a);
  return 0;
}
