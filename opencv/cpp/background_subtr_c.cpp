//-------------------------------------------------------------------------------------------
/*! \file    background_subtr_c.cpp
    \brief   Apply background subtraction,
             and find if a moving object is a big one around the image center
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.23, 2017

g++ -I -Wall background_subtr_c.cpp -o background_subtr_c.out -lopencv_core -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_highgui -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  const char *window("Background Subtraction");
  cv::namedWindow(window,1);
  int history(30);
  cv::createTrackbar( "History:", window, &history, 100, NULL);
  // cv::namedWindow("camera",1);
  // cv::namedWindow("mask",1);

  cv::Ptr<cv::BackgroundSubtractorMOG2> bkg_sbtr= cv::createBackgroundSubtractorMOG2(history, /*varThreshold=*/10.0, /*detectShadows=*/true);

  cv::Mat frame, mask, mask_f, frame_masked;
  bool running(true);
  for(int i(0);;++i)
  {
    if(running)
    {
      cap >> frame; // get a new frame from camera

      // cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
      bkg_sbtr->apply(frame,mask,1./float(history));

      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 5);
      // cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);

      mask_f= cv::Mat::zeros(mask.rows+2, mask.cols+2, CV_8UC1);
      cv::Scalar newVal(100,100,100);
      int lo(20), up(20);
      int newMaskVal= 255;
      int connectivity= 8;
      int flags= connectivity + (newMaskVal << 8 ) + cv::FLOODFILL_FIXED_RANGE + cv::FLOODFILL_MASK_ONLY;
      cv::floodFill(mask, mask_f, cv::Point(mask.cols/2,mask.rows/2), newVal, 0, cv::Scalar(lo,lo,lo ), cv::Scalar(up,up,up), flags);
      mask_f= mask_f(cv::Range(1, mask_f.rows-1), cv::Range(1, mask_f.cols-1));
      if(cv::countNonZero(mask_f)>0.5*mask.cols*mask.rows)
        mask_f.setTo(0);

      // frame_masked= cv::Scalar(0.0,0.0,0.0);
      // frame.copyTo(frame_masked, mask);
      frame_masked= 0.3*frame;
      cv::Mat masks[3]= {mask, 0.5*mask, 0.5*mask_f}, cmask;
      cv::merge(masks,3,cmask);
      frame_masked+= cmask;

      // cv::imshow(window, frame);
      cv::imshow(window, frame_masked);
      // cv::imshow("mask", mask);
    }

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  running= !running;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
