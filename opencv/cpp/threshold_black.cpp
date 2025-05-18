//-------------------------------------------------------------------------------------------
/*! \file    threshold_black.cpp
    \brief   Detect black colors.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2017

g++ -g -Wall -O2 -o threshold_black.out threshold_black.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
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


int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("camera", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("detected", cv::WINDOW_AUTOSIZE);

  int thresh_h(180), thresh_s(255), thresh_v(13);
  cv::createTrackbar("thresh_h", "camera", &thresh_h, 255, NULL);
  cv::createTrackbar("thresh_s", "camera", &thresh_s, 255, NULL);
  cv::createTrackbar("thresh_v", "camera", &thresh_v, 255, NULL);

  for(int i(0);;++i)
  {
    cv::Mat frame, frame_hsv, frame_black;
    cap >> frame;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    // cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // "THRESH_: \n 0: BINARY \n 1: BINARY_INV \n 2: TRUNC \n 3: TOZERO \n 4: TOZERO_INV";
    // cv::threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

    cv::inRange(frame_hsv, cv::Scalar(0, 0, 0), cv::Scalar(thresh_h, thresh_s, thresh_v), frame_black);

    // modified= 0.5*src;
    // cv::Mat maskbgr[3]= {dst*128.0,dst*128.0,dst*0.0}, maskcol;
    // cv::merge(maskbgr,3,maskcol);
    // modified+= maskcol;
    cv::imshow("camera", frame);
    cv::imshow("detected", frame_black);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
