//-------------------------------------------------------------------------------------------
/*! \file    cv2-resize.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.18, 2017

g++ -g -Wall -O2 -o cv2-resize.out cv2-resize.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "cap_open.h"
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("camera", cv::WINDOW_AUTOSIZE);

  int type1(1), type2(1);
  // 0 INTER_NEAREST
  // 1 INTER_LINEAR
  // 2 INTER_AREA
  // 3 INTER_CUBIC
  // 4 INTER_LANCZOS4
  int size_w(3), size_h(3);
  cv::createTrackbar("type1", "camera", &type1, 4, NULL);
  cv::createTrackbar("type2", "camera", &type2, 4, NULL);
  cv::createTrackbar("size_w", "camera", &size_w, 10, NULL);
  cv::createTrackbar("size_h", "camera", &size_h, 10, NULL);

  cv::Mat frame, frame_gray, small, large;
  while(true)
  {
    cap >> frame;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::resize(frame_gray, small, cv::Size(size_w,size_h), 0, 0, type1);
    cv::resize(small, large, frame_gray.size(), 0, 0, type2);

    cv::Mat disps[3]= {0.5*frame_gray+0.5*large, 0.5*frame_gray, 0.5*frame_gray}, disp;
    cv::merge(disps,3,disp);

    cv::imshow("camera", disp);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
