//-------------------------------------------------------------------------------------------
/*! \file    mocap1.cpp
    \brief   Color-based motion capture (tracking single object).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.28, 2017

g++ -g -Wall -O2 -o mocap1.out mocap1.cpp -lopencv_imgproc -lopencv_core -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <ctime>  // clock_gettime
#include <sys/time.h>  // gettimeofday
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

// Find the largest contour and return info. bin_src should be a binary image.
// WARNING: bin_src is modified.
cv::Moments FindLargestContour(const cv::Mat &bin_src)
{
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(bin_src,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
  if(contours.size()==0)  return cv::Moments();
  double a(0.0),a_max(0.0), i_max(0);
  for(int i(0),i_end(contours.size()); i<i_end; ++i)
  {
    a= cv::contourArea(contours[i],false);
    if(a>a_max)  {a_max= a;  i_max= i;}
  }
  return cv::moments(contours[i_max]);
}
//-------------------------------------------------------------------------------------------

inline long long GetCurrentTime(void)
{
  timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  long long t(start.tv_sec*1e9);
  t+= start.tv_nsec;
  return t;
  // struct timeval time;
  // gettimeofday (&time, NULL);
  // long long t(time.tv_sec*1e9);
  // t+= time.tv_usec*1000L;
  // return t;
}
//-------------------------------------------------------------------------------------------

int H(0), S(0), V(0);

void OnMouse(int event, int x, int y, int flags, void *data)
{
  if(event == cv::EVENT_LBUTTONDOWN)
  {
    cv::Mat &frame(*reinterpret_cast<cv::Mat*>(data));
    const cv::Vec3b &p(frame.at<cv::Vec3b>(y,x));
    H= p[0]; S= p[1]; V= p[2];
  }
}

int main(int argc, char**argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::Mat src, hsv, obj;

  cv::namedWindow("camera", 1);
  cv::setMouseCallback("camera", OnMouse, &hsv);
  int dh(30), ds(30), dv(30);
  cv::createTrackbar("dh", "camera", &dh, 100, NULL);
  cv::createTrackbar("ds", "camera", &ds, 100, NULL);
  cv::createTrackbar("dv", "camera", &dv, 100, NULL);

  while(1)
  {
    cap >> src;

    // cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    src.copyTo(hsv);
    cv::inRange(hsv, cv::Scalar(H-dh,S-ds,V-dv), cv::Scalar(H+dh,S+ds,V+dv), obj);

    // cv::Moments mu= cv::moments(obj, /*binaryImage=*/true);
    cv::Moments mu= FindLargestContour(obj);

    cv::Mat img_disp;
    img_disp= 0.3*src;
    cv::Mat detecteds[3]= {255.0*obj,0.0*obj,0.0*obj}, detectedc;
    cv::merge(detecteds,3,detectedc);
    img_disp+= detectedc;

    if(mu.m00>0.0)
    {
      cv::Point2f center(mu.m10/mu.m00, mu.m01/mu.m00);
      float angle= 0.5 * std::atan2(2.0*mu.mu11/mu.m00, (mu.mu20/mu.m00 - mu.mu02/mu.m00));
      std::cout<<GetCurrentTime()<<"  "<<mu.m00<<"  "<<angle/**180.0/M_PI*/<<"  "<<center.x<<" "<<center.y<<std::endl;
      cv::line(img_disp,
            center-50.0*cv::Point2f(std::cos(angle),std::sin(angle)),
            center+50.0*cv::Point2f(std::cos(angle),std::sin(angle)),
            cv::Scalar(0,0,255,128),10,8,0);
    }

    imshow("camera", img_disp);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
