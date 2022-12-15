//-------------------------------------------------------------------------------------------
/*! \file    concat1.cpp
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
  int array1[4]={1,2,3,4}, array2[4]={5,6,7,8};
  {
    typedef Eigen::Matrix<int,1,4> Vector4;
    Vector4 v1(array1), v2(array2);
    cout<<"v1= "<<v1<<endl;
    cout<<"v2= "<<v2<<endl;
    Eigen::Matrix<int,2,4> cat_v;
    cat_v<<v1,v2;
    Eigen::Matrix<int,1,8> cat_h;
    cat_h<<v1,v2;
    cout<<"cat_v= "<<cat_v<<endl;
    cout<<"cat_h= "<<cat_h<<endl;
  }
  {
    typedef Eigen::Matrix<int,4,1> Vector4;
    Vector4 v1(array1), v2(array2);
    cout<<"v1= "<<v1<<endl;
    cout<<"v2= "<<v2<<endl;
    Eigen::Matrix<int,4,2> cat_h;
    cat_h<<v1,v2;
    Eigen::Matrix<int,8,1> cat_v;
    cat_v<<v1,v2;
    cout<<"cat_h= "<<cat_h<<endl;
    cout<<"cat_v= "<<cat_v<<endl;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
