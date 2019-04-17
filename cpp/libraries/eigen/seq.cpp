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
  for(int i(0);i<m.size();++i)  cout<<" "<<m(i);
  cout<<endl;
  return 0;
}
