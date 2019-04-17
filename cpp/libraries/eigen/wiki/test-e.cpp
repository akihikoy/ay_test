#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;
int main()
{
  MatrixXd m(2,4);
  m<< 1.0,2.0,3.0,4.0,
      0.0,-1.0,-2.0,-3.0;
  cout<<m<<endl;
  return 0;
}
