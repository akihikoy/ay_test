#include <Eigen/Dense>
#include <iostream>
int main(int argc, char **argv)
{
  using namespace std;
  using namespace Eigen;

  MatrixXd m(2,3);
  m<<1,2,3,
    4,5,6;
  cout<<m<<endl;
  // cout<<m.Zero()<<endl; // NG
  cout<<m.Zero(2,3)<<endl;
  return 0;
}
