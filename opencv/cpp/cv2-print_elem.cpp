//-------------------------------------------------------------------------------------------
/*! \file    cv2-print_elem.cpp
    \brief   Template function to print an element of a matrix.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.24, 2022

g++ -g -Wall -O2 -o cv2-print_elem.out cv2-print_elem.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>
//-------------------------------------------------------------------------------------------
std::string GetPixelVal(const cv::Mat &m, int x, int y)
{
  std::stringstream ss;
  if(m.type()==CV_8UC1)        ss<<(int)m.at<unsigned char>(y,x);
  else if(m.type()==CV_8SC1)   ss<<(int)m.at<char>(y,x);
  else if(m.type()==CV_8UC3)   ss<<m.at<cv::Vec3b>(y,x);
  else if(m.type()==CV_16UC1)  ss<<m.at<unsigned short>(y,x);
  else if(m.type()==CV_16SC1)  ss<<m.at<short>(y,x);
  else if(m.type()==CV_32FC1)  ss<<m.at<float>(y,x);
  else if(m.type()==CV_32FC3)  ss<<m.at<cv::Vec3f>(y,x);
  else  ss<<"unknown type";
  return ss.str();
}
//-------------------------------------------------------------------------------------------

#ifndef LIBRARY
int main(int argc, char**argv)
{
  #define GEN_MAT(TYPE)  cv::Mat m_##TYPE=cv::Mat::ones(5,5,TYPE);
  GEN_MAT(CV_8UC1)
  GEN_MAT(CV_8SC1)
  GEN_MAT(CV_8UC3)
  GEN_MAT(CV_16UC1)
  GEN_MAT(CV_16SC1)
  GEN_MAT(CV_32FC1)
  #define PRINT(TYPE)  std::cout<<"Matrix m_" #TYPE ".at(2,2)="<<GetPixelVal(m_##TYPE,2,2)<<std::endl;
  PRINT(CV_8UC1)
  PRINT(CV_8SC1)
  PRINT(CV_8UC3)
  PRINT(CV_16UC1)
  PRINT(CV_16SC1)
  PRINT(CV_32FC1)
  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
