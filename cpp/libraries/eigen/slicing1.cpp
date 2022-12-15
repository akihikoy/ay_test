//-------------------------------------------------------------------------------------------
/*! \file    slicing1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.15, 2022
*/
//-------------------------------------------------------------------------------------------
#include<Eigen/Core>
#include<iostream>
using namespace Eigen;
using namespace std;

int main()
{
  int array[7]= {1,2,3,4,5,6,7};
  cout<<"array= "; for(int i(0);i<7;++i)cout<<" "<<array[i]; cout<<endl;
  typedef Eigen::Matrix<int,1,7> Vector7;
  Map<Vector7> mvec(array);  // Map to share the memory with array.
  cout<<"mvec= "<<mvec<<endl;
//   mvec[seq(2,4)]*= 10;  // NOTE: Not supported in Eigen 3.3.4.
  mvec.segment(2,2)*= 10;
  mvec.block(0,2,1,2)*= 10;
  cout<<"mvec= "<<mvec<<endl;
  cout<<"array= "; for(int i(0);i<7;++i)cout<<" "<<array[i]; cout<<endl;
}
//-------------------------------------------------------------------------------------------
