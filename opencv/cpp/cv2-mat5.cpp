//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat5.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.27, 2016

g++ -g -Wall -O2 -o cv2-mat5.out cv2-mat5.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"="<<std::endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void GeneratePoints1(cv::Mat &l_points3d)
{
  int N= 10;
  double rad=0.05, height=0.06;
  double dth(2.0*M_PI/(double)N);
  l_points3d.create(N,3,CV_32F);
  for(int i(0);i<N;++i)
  {
    l_points3d.at<float>(i,0)= rad*std::cos((double)i*dth);
    l_points3d.at<float>(i,1)= height;
    l_points3d.at<float>(i,2)= rad*std::sin((double)i*dth);
  }
}

void GeneratePoints2(cv::Mat &l_points3d)
{
  int N= 10;
  double rad=0.05, height=0.06;
  double dth(2.0*M_PI/(double)N);
  l_points3d.create(N,3,CV_32F);
  for(int i(0);i<N;++i)
  {
    // Some of following work, but not a regular way
    // float *p(l_points3d.at<float*>(i));  // NOT WORK
    float *p(l_points3d.at<float[3]>(i));  // WORKS
    // cv::Vec<float,3> &p(l_points3d.at<cv::Vec<float,3> >(i));  // WORKS
    // cv::Vec3f &p(l_points3d.at<cv::Vec3f>(i));  // WORKS
    p[0]= rad*std::cos((double)i*dth);
    p[1]= rad*std::sin((double)i*dth);
    p[2]= height;
  }
}

int main(int argc, char**argv)
{
  cv::Mat points;
  GeneratePoints1(points);
  print(points);
  GeneratePoints2(points);
  print(points);
  return 0;
}
//-------------------------------------------------------------------------------------------
