//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-farn1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016

g++ -I -Wall -O2 optical-flow-farn1.cpp -o optical-flow-farn1.out -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
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

int main(int argc, char**argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("camera",1);
  cv::Mat frame_in, frame, frame_old;
  cap >> frame;
  cv::cvtColor(frame,frame,CV_BGR2GRAY);
  for(int i(0);;++i)
  {
    frame.copyTo(frame_old);
    cap >> frame_in;
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);

    // medianBlur(frame, frame, 9);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(frame_old, frame, flow,
      /*pyr_scale*/0.5, /*levels*/3, /*winsize*/2, /*iterations*/1,
      /*poly_n*/1, /*poly_sigma*/1.5, /*flags*/0);

    // visualization
    frame_in*= 0.5;
    {
      cv::Scalar col;
      float vx,vy,spd,angle;
      int dx,dy;
      const float dt(0.1);
      int step(1);
      for (int i(step); i<frame_in.cols-step; i+=step)
      {
        for (int j(step); j<frame_in.rows-step; j+=step)
        {
          const cv::Point2f &fxy=flow.at<cv::Point2f>(j,i);  // Index order is y,x
          vx= fxy.x;
          vy= fxy.y;
          spd= std::sqrt(vx*vx+vy*vy);
          if(spd<1.0 || 1000.0<spd)  continue;
          angle= std::atan2(vy,vx);
          col= CV_RGB(0.0,255.0*std::fabs(std::cos(angle)),255.0*std::fabs(std::sin(angle)));
          cv::line(frame_in, cv::Point(i,j), cv::Point(i,j)+cv::Point(dt*vx,dt*vy), col, 1);
        }
      }
    }

    cv::imshow("camera", frame_in);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
