#include<Eigen/Core>
#include<iostream>
#include<valarray>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

int main()
{
  int array[9];
  for(int i = 0; i < 9; ++i) array[i] = i;
  cout << Map<Matrix3i>(array) << endl;
  valarray<double> varray(2.1,9);
  Map<MatrixXd> mv(&varray[0],3,3);
  mv(0,0)=10;
  mv(1,1)=10;
  mv(2,2)=10;
  cout << mv << endl;
  for(int i = 0; i < 9; ++i) cout<<" "<<varray[i];
  cout<<endl;
  cout << mv.inverse() << endl;
  cout << mv.inverse()*mv << endl;
}