//-------------------------------------------------------------------------------------------
/*! \file    frame_window.cpp
    \brief   Memorize newest N frames for image processing.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.09, 2017

g++ -g -Wall -O2 -o frame_window.out frame_window.cpp -lopencv_core -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("current",1);
  cv::namedWindow("previous",1);
  cv::Mat frame;
  std::map<int,cv::Mat> history;
  for(int i(0);;++i)
  {
    int N= 20;
    cap >> frame;
    frame.copyTo(history[i]);
    cv::imshow("current", frame);
    cv::imshow("previous", history[((i-N)>=0?(i-N):0)]);
    if(i>N)  history.erase(i-N-1);
    std::cerr<<i<<" "<<history.size()<<std::endl;
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
