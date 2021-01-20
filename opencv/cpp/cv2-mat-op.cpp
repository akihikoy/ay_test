//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat-op.cpp
    \brief   cv::Mat operations
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021

g++ -g -Wall -O2 -o cv2-mat-op.out cv2-mat-op.cpp -lopencv_core && ./cv2-mat-op.out
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

#define init(_m, ...)  _m= (cv::Mat_<double>(_m.rows,_m.cols)<<__VA_ARGS__)

int main(int argc, char**argv)
{
  cv::Mat_<double> v1(1,3), v2(1,3);
  init(v1, 1,0,1);
  init(v2, 1,1,0);
  print(v1);
  print(v2);
  print(cv::norm(v1));
  print(cv::norm(v2));
  print(v1.dot(v2));
  print(v1.cross(v2));
  print("========");

  cv::Mat_<double> v3(3,1), v4(3,1);
  init(v3, 0,1,1);
  init(v4, 1,-2,3);
  print(v3);
  print(v4);
  print(cv::norm(v3));
  print(cv::norm(v4));
  print(v3.dot(v4));
  print(v3.cross(v4));
  print("========");

  cv::Point3d p1(v2);
  print(v1);
  print(p1);
  print(cv::norm(v1));
  print(cv::norm(p1));
  // print(v1.dot(p1));  NOTE: Does not work.
  // print(v1.cross(p1));  NOTE: Does not work.
  print(cv::Mat(p1).t());
  print(v1.dot(cv::Mat(p1).t()));
  print(v1.cross(cv::Mat(p1).t()));
  print("========");

  return 0;
}
//-------------------------------------------------------------------------------------------
