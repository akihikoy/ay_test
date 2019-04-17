//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.10, 2012
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
using namespace std;
#define print(var) std::cout<<#var"= "<<endl<<(var)<<std::endl

int main(int argc, char**argv)
{
  cv::Mat m1(3,3,CV_8UC1), m2(3,3,CV_8UC1), m3;
  m1= (cv::Mat_<unsigned char>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  m2= (cv::Mat_<unsigned char>(3,3)<<0,1,1, 1,0,1, 0,1,0);
  print(m1);
  print(m2);
  print(m1-m2);
  print(m1&m2);
  bitwise_xor(m1,m2,m3);
  print(m3);
  print(sum(m3)[0]);
  return 0;
}
//-------------------------------------------------------------------------------------------
