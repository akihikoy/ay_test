//-------------------------------------------------------------------------------------------
/*! \file    demo2.cpp
    \brief   Demo(2) of SLIC superpixel segmentation.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.24, 2018

g++ -g -Wall -O2 -o slic_demo2.out demo2.cpp slic.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

Usage:
./slic_demo2.out 0 320 240 200
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
#include <unistd.h>
#include "cap_open.h"
#include "slic.h"
//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  int numSuperpixel= ((argc>4)?atoi(argv[4]):100);
  SLIC slic;

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    slic.GenerateSuperpixels(frame, numSuperpixel);
    cv::Mat result= slic.GetImgWithContours(cv::Scalar(0, 0, 255));

    cv::imshow("camera", result);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
