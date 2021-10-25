//-------------------------------------------------------------------------------------------
/*! \file    find_black_contours.cpp
    \brief   Detect small black-colored things as contours.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2017

g++ -g -Wall -O2 -o find_black_contours.out find_black_contours.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

Based on:
  threshold_black.cpp
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

  cv::namedWindow("camera", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("detected", CV_WINDOW_AUTOSIZE);

  int thresh_h(180), thresh_s(255), thresh_v(13);
  cv::createTrackbar("thresh_h", "detected", &thresh_h, 255, NULL);
  cv::createTrackbar("thresh_s", "detected", &thresh_s, 255, NULL);
  cv::createTrackbar("thresh_v", "detected", &thresh_v, 255, NULL);
  int n_erode1(2), n_dilate1(2);
  cv::createTrackbar("n_dilate1", "detected", &n_dilate1, 10, NULL);
  cv::createTrackbar("n_erode1", "detected", &n_erode1, 10, NULL);
  int area_min(6), area_max(90);
  cv::createTrackbar("area_min", "detected", &area_min, 150, NULL);
  cv::createTrackbar("area_max", "detected", &area_max, 150, NULL);

  for(int i(0);;++i)
  {
    cv::Mat frame, frame_hsv, frame_black;
    cap >> frame;

    // Black detection
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(frame_hsv, cv::Scalar(0, 0, 0), cv::Scalar(thresh_h, thresh_s, thresh_v), frame_black);

    cv::dilate(frame_black,frame_black,cv::Mat(),cv::Point(-1,-1), n_dilate1);
    cv::erode(frame_black,frame_black,cv::Mat(),cv::Point(-1,-1), n_erode1);

    // Make image for display
    cv::Mat img_disp;
    img_disp= 0.3*frame;
    cv::Mat frame_blacks[3]= {255.0*frame_black,128.0*frame_black,0.0*frame_black}, frame_blackc;
    cv::merge(frame_blacks,3,frame_blackc);
    img_disp+= frame_blackc;

    // Contour detection
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(frame_black, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if(contours.size()>0)
    {
      for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      {
        cv::drawContours(img_disp, contours, ic, CV_RGB(255,0,255), /*thickness=*/2, /*linetype=*/8);
        double area= cv::contourArea(contours[ic],false);
        if(area<area_min || area_max<area)  continue;
        cv::Rect bound= cv::boundingRect(contours[ic]);
        cv::rectangle(img_disp, bound, cv::Scalar(0,0,255), 2);
      }
    }

    cv::imshow("camera", frame);
    cv::imshow("detected", img_disp);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
