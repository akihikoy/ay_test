//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.10, 2012

    compile: x++ -cv cv2-mat2.cpp -- -lopencv_core
*/
//-------------------------------------------------------------------------------------------
// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif
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
#define print(var) std::cout<<#var"= "<<endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat_<double> m1(3,3);
  m1= (cv::Mat_<double>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  cv::Vec<double,3> v1;
  v1= cv::Vec<double,3>(1,2,3);
  print(m1);
  print(cv::Mat(v1));
  print(m1*cv::Mat(v1));

  cv::Vec<double,9> vm2;
  cv::Mat_<double> m2(3,3,vm2.val);
  m1= (cv::Mat_<double>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  cv::Mat(m1.inv()).copyTo(m2);
  print(cv::Mat(vm2));
  print(m2);
  print(m2*2.0);
  // print(m2*cv::Vec3b(3,1,0));
  print(m2*cv::Mat(v1));
  cv::Mat(m2*cv::Mat(v1)).copyTo(v1);
  print(cv::Mat(v1));

  cv::Vec<double,3> v3(3,2,1);
  print(cv::Mat(v1));
  print(cv::Mat(v3));
  print(cv::Mat(v1+v3));
  print(cv::Mat(v1*2.0));
  print(norm(v1));
  print(cv::Mat(v1*(1.0/norm(v1))));
  cv::normalize(v1,v1);
  print(cv::Mat(v1));

  cv::Matx<double,3,3> m3(1,2,3, 4,5,6, 7,8,9);
  m3<<1,2,3, 4,5,6, 7,8,9;
  print(cv::Mat(m3));
  print(cv::Mat(v3));
  print(cv::Mat(m3*v3));
  v3= cv::Mat(m3*v3);
  print(cv::Mat(v3));

  // elemental access
  print(cv::Mat(v1));
  print(m1);
  print(v1(0));
  print(v1(1));
  print(m1(0,0));
  print(m1(0,1));
  print(m1(0,2));

  return 0;
}
//-------------------------------------------------------------------------------------------
