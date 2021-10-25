//-------------------------------------------------------------------------------------------
/*! \file    find_white_contours.cpp
    \brief   Detect white-colored things as contours.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2017

g++ -g -Wall -O2 -o find_white_contours.out find_white_contours.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

Based on:
  (copied from:) find_black_contours.cpp
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

/*Find contours of white areas.
  frame: Input image.
  frame_white: Detected white image.
  contours: Found contours.
  v_min, s_max: Thresholds of V-minimum and S-maximum of HSV.
  n_dilate, n_erode: dilate and erode filter parameters before detecting contours.
*/
void FindWhiteContours(
    const cv::Mat &frame,
    cv::Mat &frame_white,
    std::vector<std::vector<cv::Point> > &contours,
    int v_min=100, int s_max=20, int n_dilate=1, int n_erode=1)
{
  cv::Mat frame_hsv;

  // White detection
  cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
  cv::inRange(frame_hsv, cv::Scalar(0, 0, v_min), cv::Scalar(255, s_max, 255), frame_white);

  if(n_dilate>0)  cv::dilate(frame_white,frame_white,cv::Mat(),cv::Point(-1,-1), n_dilate);
  if(n_erode>0)   cv::erode(frame_white,frame_white,cv::Mat(),cv::Point(-1,-1), n_erode);

  // Contour detection
  cv::findContours(frame_white, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}
//-------------------------------------------------------------------------------------------

// Make a mask from biggest contour.
void MakeBiggestContourMask(const std::vector<std::vector<cv::Point> > &contours, cv::Mat &mask, int fill_value=1)
{
  if(contours.size()==0)  return;
  double a(0.0),a_max(0.0), i_max(0);
  for(int i(0),i_end(contours.size()); i<i_end; ++i)
  {
    a= cv::contourArea(contours[i],false);
    if(a>a_max)  {a_max= a;  i_max= i;}
  }
  cv::drawContours(mask, contours, i_max, fill_value, /*thickness=*/-1);
}
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("camera", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("detected", CV_WINDOW_AUTOSIZE);

  int /*thresh_h(0), */thresh_s(20), thresh_v(100);
  // cv::createTrackbar("thresh_h", "detected", &thresh_h, 255, NULL);
  cv::createTrackbar("thresh_s", "detected", &thresh_s, 255, NULL);
  cv::createTrackbar("thresh_v", "detected", &thresh_v, 255, NULL);
  int n_erode1(1), n_dilate1(1);
  cv::createTrackbar("n_dilate1", "detected", &n_dilate1, 10, NULL);
  cv::createTrackbar("n_erode1", "detected", &n_erode1, 10, NULL);
  int area_min(1000)/*, area_max(90)*/;
  cv::createTrackbar("area_min", "detected", &area_min, 10000, NULL);
  // cv::createTrackbar("area_max", "detected", &area_max, 150, NULL);

  for(int i(0);;++i)
  {
    cv::Mat frame, frame_white;
    cap >> frame;

    std::vector<std::vector<cv::Point> > contours;
    FindWhiteContours(frame, frame_white, contours,
          /*v_min=*/thresh_v, /*s_max=*/thresh_s, /*n_dilate=*/n_dilate1, /*n_erode=*/n_erode1);

    // Make a mask of biggest contour:
    cv::Mat mask_biggest(frame_white.size(), CV_8UC1);
    mask_biggest.setTo(0);
    MakeBiggestContourMask(contours, mask_biggest);

    // Make image for display
    cv::Mat img_disp;
    img_disp= 0.3*frame;
    cv::Mat frame_whites[3]= {128.0*frame_white,128.0*frame_white,0.0*frame_white+200.0*mask_biggest}, frame_whitec;
    cv::merge(frame_whites,3,frame_whitec);
    img_disp+= frame_whitec;

    // Draw contours
    if(contours.size()>0)
    {
      for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      {
        double area= cv::contourArea(contours[ic],false);
        if(area<area_min /*|| area_max<area*/)  continue;
        cv::drawContours(img_disp, contours, ic, CV_RGB(255,0,255), /*thickness=*/2, /*linetype=*/8);

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
