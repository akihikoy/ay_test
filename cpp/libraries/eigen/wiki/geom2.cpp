#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  Affine3d T;
  T= Translation3d(0.5,0.0,0.0) * AngleAxisd(0.8,Vector3d::UnitZ());
  print(T.matrix());
  print(T.linear());
  print(T.translation());

  return 0;
}
