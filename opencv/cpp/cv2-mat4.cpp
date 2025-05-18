//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat4.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2016

g++ -g -Wall -O2 -o cv2-mat4.out cv2-mat4.cpp -lopencv_core -I/usr/include/opencv4
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
  cv::Mat_<float> m1(3,3);
  // cv::Mat m1(3,3,CV_32F);
  m1= (cv::Mat_<float>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  cv::Vec3f v1(-1.0,0.0,2.0);
  print(m1);
  print(cv::Mat(v1));

  for(int r(0),r_end(m1.rows);r<r_end;++r)
  {
    // m1.row(r)= m1.row(r) + v1;
    m1.row(r)+= cv::Mat(v1).t();
  }
  print(m1);

  print(m1.at<float>(0,2));
  print(m1(cv::Rect(1,0,2,2)));
  print(m1(cv::Rect(2,0,1,2)));

  return 0;
}
//-------------------------------------------------------------------------------------------
