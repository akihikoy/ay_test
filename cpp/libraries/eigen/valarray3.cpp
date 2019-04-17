#include<Eigen/Core>
#include<iostream>
#include<valarray>
using namespace Eigen;
using namespace std;

// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main()
{
  int array[9];
  for(int i = 0; i < 9; ++i) array[i] = i+1;
  print(Map<VectorXi>(array,9).transpose());
  print(Map<Matrix3i>(array));
  print(Map<MatrixXi>(array,2,3));
  Map<VectorXi> mv(array,9);
  Map<MatrixXi> mm(array,2,3);
  mv(3)= 0;
  print(Map<VectorXi>(array,9).transpose());
  mm(0,2)= 0;
  print(mm);
  print(Map<VectorXi>(array,9).transpose());
  return 0;
}
