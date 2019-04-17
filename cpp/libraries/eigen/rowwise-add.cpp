#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl
int main()
{
  MatrixXd m(4,3);
  m<< 1,1,1,
      1,2,3,
      2,1,2,
      0,0,1;
  print(m);
  print(m.rowwise() + RowVector3d(-1,-2,-1));
  return 0;
}
