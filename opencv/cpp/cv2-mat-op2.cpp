//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat-op2.cpp
    \brief   cv::Mat operations.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.01, 2022

g++ -g -Wall -O2 -o cv2-mat-op2.out cv2-mat-op2.cpp -lopencv_core -I/usr/include/opencv4 && ./cv2-mat-op2.out
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
#define print(var) std::cout<<#var"= "<<std::endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

template<typename t_elem> int CvMatType();
template<> int CvMatType<unsigned char>()  {return CV_8UC1;}
template<> int CvMatType<float>()  {return CV_32FC1;}

int main(int argc, char**argv)
{
  // typedef unsigned char t_elem;
  typedef float t_elem;
  cv::Mat m1(3,3,CvMatType<t_elem>()), m2(3,3,CvMatType<t_elem>()), m3;
  m1= (cv::Mat_<t_elem>(3,3)<<1,2,3, 4,5,6, 7,8,9);
  m2= (cv::Mat_<t_elem>(3,3)<<0,1,2, 1,2,3, 3,4,5);
  print(m1);
  print(m2);
  print(m1+m2);
  print(m1-3);
  if(CvMatType<t_elem>()==CV_32FC1)
    print(m1*m2);  // NOTE: This works with float and double mat only.
  print(m1.mul(m2));
  print((m1-3).mul(m2/2));
  return 0;
}
//-------------------------------------------------------------------------------------------
