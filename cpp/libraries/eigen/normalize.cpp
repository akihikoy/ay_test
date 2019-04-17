#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  Vector3d v(1,2,3);
  print(v);
  print(v.norm());
  v.normalize();
  print(v);

  return 0;
}
