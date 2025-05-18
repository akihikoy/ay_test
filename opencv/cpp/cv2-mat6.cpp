//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat6.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2016

g++ -g -Wall -O2 -o cv2-mat6.out cv2-mat6.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
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
  cv::Mat_<float> m1(3,3), m2;
  // cv::Mat m1(3,3,CV_32F);
  m1= (cv::Mat_<float>(3,3)<<1.5,0,1, 0,0,1, 0,5.5,1);
  cv::Vec3f v1(-1.5,0.0,2.0);
  print(m1);
  print(cv::Mat(v1));
  print(cv::Mat(v1).t());

  cv::flip(m1,m2,0);
  print(m2);

  print(m1.type());
  // print(v1.type());
  print(cv::Mat(v1).type());

  // m1.push_back(v1);  // ERROR: cv::Exception
  // print(m1);
  // m1.push_back(cv::Mat(v1).t());  // ERROR: cv::Exception
  // print(m1);
  cv::Mat v2(cv::Mat(v1).t());
  m1.push_back(v2);
  print(m1);

  cv::Point3f p1(-1.5,0.0,2.5);
  print(p1);
  // m1.push_back(p1);  // ERROR: cv::Exception
  // print(m1);
  cv::Mat v3(p1);
  print(v3);
  v3= v3.t();
  m1.push_back(v3);
  print(m1);

  return 0;
}
//-------------------------------------------------------------------------------------------
