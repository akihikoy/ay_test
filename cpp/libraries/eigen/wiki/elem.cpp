#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う
using namespace std;

int main(int,char**)
{
  MatrixXd m(2,2);
  m(0,0) = 3.0;
  m(1,0) = 2.5;
  m(0,1) = -1.0;
  m(1,1) = m(1,0) + m(0,1);
  cout<<"0,0: "<<m(0,0)<<endl;
  cout<<"1,0: "<<m(1,0)<<endl;
  cout<<"1,1: "<<m(1,1)<<endl;
  VectorXd v(3);
  v(0)= 1.0;
  v[1]= -1.0;
  v(2)= 2.0;
  cout<<"0: "<<v(0)<<endl;
  cout<<"2: "<<v[2]<<endl;
  return 0;
}
