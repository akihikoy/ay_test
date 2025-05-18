//-------------------------------------------------------------------------------------------
/*! \file    cv2-mean_stddev.cpp
    \brief   Compute mean and std-dev with OpenCV.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.23, 2023

g++ -g -Wall -O2 -o cv2-mean_stddev.out cv2-mean_stddev.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::vector<float> data_f;
  std::vector<cv::Vec3f> data_v3;
  for(int i(0); i<20; ++i)
  {
    data_f.push_back(i);
    data_v3.push_back(cv::Vec3f(i,5.0,i*10));
  }
  {
    cv::Mat mean_f,stddev_f;
    cv::Mat mean_v3,stddev_v3;
    cv::meanStdDev(data_f,mean_f,stddev_f);
    cv::meanStdDev(data_v3,mean_v3,stddev_v3);
    std::cout<<"data_f:mean,stddev(Mat): "<<mean_f<<", "<<stddev_f<<std::endl;
    std::cout<<"data_v3:mean,stddev(Mat): "<<mean_v3.t()<<", "<<stddev_v3.t()<<std::endl;
  }
  {
    cv::Scalar mean_f,stddev_f;
    cv::Scalar mean_v3,stddev_v3;
    cv::meanStdDev(data_f,mean_f,stddev_f);
    cv::meanStdDev(data_v3,mean_v3,stddev_v3);
    std::cout<<"data_f:mean,stddev(Scalar): "<<mean_f<<", "<<stddev_f<<std::endl;
    std::cout<<"data_v3:mean,stddev(Scalar): "<<mean_v3.t()<<", "<<stddev_v3.t()<<std::endl;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
