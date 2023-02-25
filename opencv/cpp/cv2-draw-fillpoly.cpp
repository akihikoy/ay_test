//-------------------------------------------------------------------------------------------
/*! \file    cv2-draw-fillpoly.cpp
    \brief   Test of fillPoly to check if the pixels on the polygon are filled.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.25, 2023

g++ -g -Wall -O2 -o cv2-draw-fillpoly.out cv2-draw-fillpoly.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat img(5, 5, CV_8UC1);
  cv::namedWindow("image",1);
  std::vector<std::vector<cv::Point> >  points(1);
  int mode(0);  //0:fillPoly, 1:polylines, 2:both.
  CreateTrackbar<int>("mode", "image", &mode, 0, 2, 1, &TrackbarPrintOnTrack);
  int pattern(0);
  CreateTrackbar<int>("pattern", "image", &pattern, 0, 2, 1, &TrackbarPrintOnTrack);
  while(1)
  {
    img.setTo(0);
    points[0].clear();
    if(pattern==0)
    {
      points[0].push_back(cv::Point(1,1));
      points[0].push_back(cv::Point(1,3));
      points[0].push_back(cv::Point(3,3));
      points[0].push_back(cv::Point(3,1));
    }
    else if(pattern==1)
    {
      points[0].push_back(cv::Point(1,1));
      points[0].push_back(cv::Point(1,3));
      points[0].push_back(cv::Point(3,1));
    }
    else if(pattern==2)
    {
      points[0].push_back(cv::Point(1,1));
      points[0].push_back(cv::Point(1,2));
      points[0].push_back(cv::Point(3,1));
    }
    if(mode==0 || mode==2)  cv::fillPoly(img, points, cv::Scalar(255));
    if(mode==1 || mode==2)  cv::polylines(img, points, /*isClosed=*/true, cv::Scalar(255), 1);

    cv::Mat disp;
    cv::resize(img, disp, cv::Size(), 30, 30, cv::INTER_NEAREST);
    cv::imshow("image", disp);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
