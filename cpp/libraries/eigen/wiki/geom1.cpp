#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  double q=0.8;
  Matrix4d rot1;
  rot1<<
      cos(q), -sin(q),  0, 0,
      sin(q),  cos(q),  0, 0,
           0,       0,  1, 0,
           0,       0,  0, 1;
  print(rot1);

  Matrix4d trans1;
  trans1<<
      1,0,0, 0.5,
      0,1,0, 0,
      0,0,1, 0,
      0,0,0, 1;
  print(trans1);
  print(trans1*rot1);

  cout<<"\n--------\n"<<endl;

  Affine3d rot2;
  rot2= AngleAxisd(q,Vector3d(0,0,1));
  print(rot2.matrix());

  Affine3d trans2;
  trans2= Translation3d(0.5,0.0,0.0);
  print(trans2.matrix());

  print((trans2*rot2).matrix());

  return 0;
}
