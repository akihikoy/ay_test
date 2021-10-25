//-------------------------------------------------------------------------------------------
/*! \file    track_black_markers.cpp
    \brief   Track black-colored markers.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2017

We use meanShift to track black markers detected by thresholding and contour detection.
We expected this approach is robust and efficient compared to global blob detection approach
(cf. simple_blob_tracker2.cpp and simple_blob_tracker3.cpp).
Actually the marker tracking was robust.
However it turned out that this approach does not give us good accuracy of marker track.
meanShift gives an updated object location as Rect which has integer values.
Since the marker movement is small (a few pixels), the movement looks jumpy.
Blob detection gives a position and a size in float type, which looks very smooth.
Therefore we improved the robustness of blob-detection-based marker tracking.
See simple_blob_tracker4.cpp

g++ -g -Wall -O2 -o track_black_markers.out track_black_markers.cpp -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui -lopencv_videoio

Based on:
  threshold_black.cpp
  find_black_contours.cpp
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
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

  int calib_phase(1);
  std::vector<cv::Rect> bounds, bounds_orig;

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
    cv::Mat frame_blacks[3]= {100.0*frame_black,60.0*frame_black,0.0*frame_black}, frame_blackc;
    cv::merge(frame_blacks,3,frame_blackc);
    img_disp+= frame_blackc;

    // Contour detection
    if(calib_phase==1)
    {
      bounds.clear();
      bounds_orig.clear();
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours(frame_black, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
      cv::drawContours(img_disp, contours, 0, CV_RGB(0,255,0), /*thickness=*/2, /*linetype=*/8);
      if(contours.size()>0)
      {
        for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
        {
          double area= cv::contourArea(contours[ic],false);
          if(area<area_min || area_max<area)  continue;
          cv::Rect bound= cv::boundingRect(contours[ic]);
          //*TEST*/cv::meanShift(frame_black, bound, cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
          bounds.push_back(bound);
          bounds_orig.push_back(bound);
          cv::rectangle(img_disp, bound, cv::Scalar(0,0,255), 2);
        }
      }
      calib_phase= 2;
    }

    if(calib_phase>=2)
    {
      for(int ib(0),ib_end(bounds.size()); ib<ib_end; ++ib)
      {
        cv::meanShift(frame_black, bounds_orig[ib], cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
        bounds[ib]= bounds_orig[ib];
      }
      ++calib_phase;
      if(calib_phase>20)  calib_phase= 0;
    }

    // Track rect bounds
    if(calib_phase==0)
    {
      for(int ib(0),ib_end(bounds.size()); ib<ib_end; ++ib)
      {
        int nitr= cv::meanShift(frame_black, bounds[ib], cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
        cv::rectangle(img_disp, bounds[ib], cv::Scalar(0,128,255), 2);
        cv::Point d(bounds[ib].x-bounds_orig[ib].x, bounds[ib].y-bounds_orig[ib].y);
        if(std::abs(d.x)<=1)  d.x= 0;
        if(std::abs(d.y)<=1)  d.y= 0;
        cv::Point center(0.5*(bounds_orig[ib].tl()+bounds_orig[ib].br()));
        cv::line(img_disp, center, center+10*d, cv::Scalar(0,0,255), 3, 8, 0);
        std::cerr<<" "<<nitr;
      }
      std::cerr<<std::endl;
    }

    cv::imshow("camera", frame);
    cv::imshow("detected", img_disp);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c=='c')  calib_phase= 1;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
