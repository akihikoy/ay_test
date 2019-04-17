#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main()
{
  Matrix3d m;
  // MatrixXd m(3,3);
  print(m);
  print(m.size());
  print(m.rows());
  print(m.cols());
  m.resize(4,4);
  print(m);
  print(m.size());
  print(m.rows());
  print(m.cols());
  return 0;
}
