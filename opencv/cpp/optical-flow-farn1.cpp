//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-farn1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016

g++ -I -Wall -O2 optical-flow-farn1.cpp -o optical-flow-farn1.out -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include "cap_open.h"
#define LIBRARY
#include "float_trackbar.cpp"
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
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  const char *window("Optical Flow Farneback");
  cv::namedWindow(window,1);

  float v_min(1.0), v_max(1000.0);
  CreateTrackbar<float>("v_min", window, &v_min, 0.0f, 100.0f, 0.1f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("v_max", window, &v_max, 0.0f, 1000.0f, 0.01f,  &TrackbarPrintOnTrack);

  double pyr_scale(0.5);
  int levels(3);
  int winsize(2);
  int iterations(1);
  int poly_n(1);
  double poly_sigma(1.5);
  int flags(0);
  int viz_step(1);

  CreateTrackbar<double>("pyr_scale:", window, &pyr_scale, 0.0, 0.99, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("levels:", window, &levels, 1, 10, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("winsize:", window, &winsize, 1, 30, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("iterations:", window, &iterations, 0, 10, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("poly_n:", window, &poly_n, 0, 10, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("poly_sigma:", window, &poly_sigma, 0.0, 10.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("flags:", window, &flags, 0, 1, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int   >("viz_step:", window, &viz_step, 1, 30, 1, &TrackbarPrintOnTrack);

  cv::Mat frame_in, frame, frame_old;
  cap >> frame;
  cv::cvtColor(frame,frame,CV_BGR2GRAY);
  for(int i(0);;++i)
  {
    frame.copyTo(frame_old);
    if(!cap.Read(frame_in))
    {
      if(cap.WaitReopen()) {i=-1; continue;}
      else break;
    }
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);

    // medianBlur(frame, frame, 9);

    cv::Mat flow;
    // cv::calcOpticalFlowFarneback(frame_old, frame, flow,
    //   /*pyr_scale*/0.5, /*levels*/3, /*winsize*/2, /*iterations*/1,
    //   /*poly_n*/1, /*poly_sigma*/1.5, /*flags*/0);
    int flag_map[]={0,cv::OPTFLOW_FARNEBACK_GAUSSIAN};
    int flags2= flag_map[flags];
    cv::calcOpticalFlowFarneback(frame_old, frame, flow,
      pyr_scale, levels, winsize, iterations,
      poly_n, poly_sigma, flags2);

    // copy optical flow to a compatible form.
    cv::Mat velx(frame.rows, frame.cols, CV_32FC1);
    cv::Mat vely(frame.rows, frame.cols, CV_32FC1);
    velx= cv::Scalar(0);
    vely= cv::Scalar(0);
    for (int i(0); i<frame.cols; ++i)
    {
      for (int j(0); j<frame.rows; ++j)
      {
        const cv::Point2f &fxy=flow.at<cv::Point2f>(j,i);
        velx.at<float>(j,i)= fxy.x;
        vely.at<float>(j,i)= fxy.y;
      }
    }

    // visualization
    frame_in*= 0.5;
    {
      cv::Scalar col;
      float vx,vy,spd,angle;
      int dx,dy;
      const float dt(0.1);
      for (int i(viz_step); i<frame_in.cols-viz_step; i+=viz_step)
      {
        for (int j(viz_step); j<frame_in.rows-viz_step; j+=viz_step)
        {
          vx= velx.at<float>(j, i);  // Index order is y,x
          vy= vely.at<float>(j, i);  // Index order is y,x
          spd= std::sqrt(vx*vx+vy*vy);
          if(spd<v_min || v_max<spd)  continue;
          angle= std::atan2(vy,vx);
          col= cv::Scalar(0.0,255.0*std::fabs(std::cos(angle)),255.0*std::fabs(std::sin(angle)));
          cv::line(frame_in, cv::Point(i,j), cv::Point(i,j)+cv::Point(dt*vx,dt*vy), col, 1);
        }
      }
    }

    cv::imshow(window, frame_in);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
