//-------------------------------------------------------------------------------------------
/*! \file    array1.cpp
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
  int array[3]= {10,20,30};
  cout<<"array= "; for(int i(0);i<3;++i)cout<<" "<<array[i]; cout<<endl;
  typedef Eigen::Matrix<int,1,3> Vector3;
  Vector3 vec(array);
  cout<<"vec= "<<vec<<endl;
  vec[1]*= 10;
  array[2]*= 10;
  cout<<"vec= "<<vec<<endl;
  cout<<"array= "; for(int i(0);i<3;++i)cout<<" "<<array[i]; cout<<endl;
  Map<Vector3> mvec(array);
  cout<<"mvec= "<<mvec<<endl;
  mvec[1]*= 10;
  array[2]*= 10;
  cout<<"mvec= "<<mvec<<endl;
  cout<<"array= "; for(int i(0);i<3;++i)cout<<" "<<array[i]; cout<<endl;
  Map<Vector3>(array)*= 100;
  cout<<"array= "; for(int i(0);i<3;++i)cout<<" "<<array[i]; cout<<endl;
}
//-------------------------------------------------------------------------------------------
