#include<Eigen/Core>
#include<iostream>
#include<valarray>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

int main()
{
  double array[9];
  for(int i = 0; i < 9; ++i) array[i] = i;
  valarray<double> varray(array,9);
  // VectorXd mv(&varray[0],9);
  Map<VectorXd> mv(&varray[0],9);
  mv(0)=10;
  mv(1)=10;
  mv(2)=10;
  cout << mv.transpose() << endl;
  for(int i = 0; i < 9; ++i) cout<<" "<<varray[i];
  cout<<endl;
}
