//-------------------------------------------------------------------------------------------
/*! \file    cv2-puttext.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.24, 2017

g++ -g -Wall -O2 -o cv2-puttext.out cv2-puttext.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
  // return ros::Time::now().toSec();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::namedWindow("time", CV_WINDOW_AUTOSIZE);
  cv::Mat frame(cv::Size(320,50),CV_8UC3);
  while(true)
  {
    double time= GetCurrentTime();
    std::stringstream ss;
    ss<<std::setprecision(14)<<time;
    frame.setTo(0);
    double font_scale(1.0);
    int thickness(1);
    cv::putText(frame, ss.str(), cv::Point(10,35), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0,255,0), thickness, CV_AA);
    cv::imshow("time", frame);
    char c(cv::waitKey(100));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
