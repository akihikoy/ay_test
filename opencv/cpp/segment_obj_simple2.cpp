//-------------------------------------------------------------------------------------------
/*! \file    segment_obj_simple2.cpp
    \brief   Segment objects on a plate of specific color (e.g. white).
             Segmentation is based on Canny edge detection.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.17, 2017

g++ -g -Wall -O2 -o segment_obj_simple2.out segment_obj_simple2.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

Based on:
  segment_obj_simple1.cpp
  cv2-canny.cpp
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
void MakeBiggestContourMask(const std::vector<std::vector<cv::Point> > &contours,
    cv::Mat &mask, bool convex=false, int fill_value=1)
{
  if(contours.size()==0)  return;
  double a(0.0),a_max(0.0), i_max(0);
  for(int i(0),i_end(contours.size()); i<i_end; ++i)
  {
    a= cv::contourArea(contours[i],false);
    if(a>a_max)  {a_max= a;  i_max= i;}
  }
  if(!convex)
    cv::drawContours(mask, contours, i_max, fill_value, /*thickness=*/-1);
  else
  {
    std::vector<std::vector<cv::Point> > hull(1);
    cv::convexHull(contours[i_max], hull[0], /*clockwise=*/true);
    cv::drawContours(mask, hull, 0, fill_value, /*thickness=*/-1);
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("detected", CV_WINDOW_AUTOSIZE);

  // For white detector:
  int white_s_max(20), white_v_min(100);
  cv::createTrackbar("white_s_max", "detected", &white_s_max, 255, NULL);
  cv::createTrackbar("white_v_min", "detected", &white_v_min, 255, NULL);
  int n_erode1(1), n_dilate1(1);
  cv::createTrackbar("n_dilate1", "detected", &n_dilate1, 10, NULL);
  cv::createTrackbar("n_erode1", "detected", &n_erode1, 10, NULL);

  int thresh_canny(40);
  cv::createTrackbar("thresh_canny", "detected", &thresh_canny, 100, NULL);
  int n_erode2(1), n_dilate2(1);
  cv::createTrackbar("n_dilate2", "detected", &n_dilate2, 10, NULL);
  cv::createTrackbar("n_erode2", "detected", &n_erode2, 10, NULL);
  int rect_len_min(40), rect_len_max(400);
  cv::createTrackbar("rect_len_min", "detected", &rect_len_min, 600, NULL);
  cv::createTrackbar("rect_len_max", "detected", &rect_len_max, 600, NULL);

  for(int i(0);;++i)
  {
    cv::Mat frame, mask_white;
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    std::vector<std::vector<cv::Point> > contours_w;
    FindWhiteContours(frame, mask_white, contours_w,
          /*v_min=*/white_v_min, /*s_max=*/white_s_max, /*n_dilate=*/n_dilate1, /*n_erode=*/n_erode1);

    // Make a mask of biggest contour:
    cv::Mat mask_biggest(mask_white.size(), CV_8UC1);
    mask_biggest.setTo(0);
    MakeBiggestContourMask(contours_w, mask_biggest);

    // Detect objects-on-white by edge
    cv::Mat frame_gray/*, frame_white, frame_white_gray*/;
    cv::Mat mask_objects;
    // frame.copyTo(frame_white, mask_biggest);

    // Edge detection
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::blur(frame_gray, frame_gray, cv::Size(3,3));
    cv::Canny(frame_gray, mask_objects, thresh_canny, thresh_canny*3, /*kernel_size=*/3);
    mask_objects.setTo(0, 1-mask_biggest);

    cv::dilate(mask_objects,mask_objects,cv::Mat(),cv::Point(-1,-1), n_dilate2);
    cv::erode(mask_objects,mask_objects,cv::Mat(),cv::Point(-1,-1), n_erode2);

    // Make image for display
    cv::Mat img_disp;
    img_disp= 0.3*frame;
    cv::Mat mask_objectss[3]= {128.0*mask_biggest,128.0*mask_biggest,128.0*mask_biggest+128.0*mask_objects}, mask_objectsc;
    cv::merge(mask_objectss,3,mask_objectsc);
    img_disp+= mask_objectsc;

    // Find object contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask_objects, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if(contours.size()>0)
    {
      for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      {
        // double area= cv::contourArea(contours[ic],false);
        cv::Rect bound= cv::boundingRect(contours[ic]);
        int bound_len= std::max(bound.width, bound.height);
        if(bound_len<rect_len_min || bound_len>rect_len_max)  continue;
        cv::drawContours(img_disp, contours, ic, CV_RGB(255,0,255), /*thickness=*/2, /*linetype=*/8);

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
