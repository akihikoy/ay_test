//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.07, 2016

    g++ -g -Wall -O2 -o cv2-mat3.out cv2-mat3.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat_<double> m1(3,4), m2;
  m1= (cv::Mat_<double>(3,4)<<0,0,1,2, 0,0,1,1, 0,1,1,0);
  cv::reduce(m1,m2,0,CV_REDUCE_MAX);
  print(m1);
  print(m1.size());
  print(m2);
  print(m2.size());

  print(m1.col(0));
  print(m1.col(3));
  print(m1.row(0));
  print(m1.row(2));
  print(cv::Vec3d(m1.col(3)));
  print(cv::Vec4d(m1.row(0)));
  return 0;
}
//-------------------------------------------------------------------------------------------
