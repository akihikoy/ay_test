//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-plk2.cpp
    \brief   Test of calcOpticalFlowPyrLK with continuous feature point detection (sparse optical flow).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.23, 2020

g++ -I -Wall -O2 optical-flow-plk2.cpp -o optical-flow-plk2.out -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <map>
#include <ctype.h>
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

  const char *window("Parameters");
  cv::namedWindow(window,1);
  cv::namedWindow("camera",1);
  int ni(1);
  float v_min(0.1), v_max(1000.0);
  CreateTrackbar<int>("Interval", window, &ni, 0, 100, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("v_min", window, &v_min, 0.0f, 100.0f, 0.1f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("v_max", window, &v_max, 0.0f, 1000.0f, 0.01f,  &TrackbarPrintOnTrack);

  cv::TermCriteria term_criteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);
  int max_feat_count(500);
  double quality_level(0.001);
  double min_distance(10);
  int block_size(3);
  bool use_harris(false);
  double harris_k(0.04);
  int subpix_win_size(10);
  int lk_win_size(31);
  int max_level(3);
  double min_eig_th(0.001);
  int optflow_flags(0);

  CreateTrackbar<int>("max_feat_count", window, &max_feat_count, 0, 10000, 10,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("quality_level", window, &quality_level, 0.0, 0.1, 0.0001,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("min_distance", window, &min_distance, 0.0, 50.0, 1.0,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("block_size", window, &block_size, 1, 51, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("use_harris", window, &use_harris, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("harris_k", window, &harris_k, 0.0, 0.1, 0.001,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("subpix_win_size", window, &subpix_win_size, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("lk_win_size", window, &lk_win_size, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("max_level", window, &max_level, 0, 10, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("min_eig_th", window, &min_eig_th, 0.0, 0.1, 0.0001,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("optflow_flags", window, &optflow_flags, 0, 1, 1,  &TrackbarPrintOnTrack);
  cv::imshow(window, cv::Mat(10, 200, CV_8UC3));

  cv::Mat frame_in, frame, frame_old;
  std::map<int,cv::Mat> history;
  for(int f(0);;++f)
  {
    if(!cap.Read(frame_in))
    {
      if(cap.WaitReopen()) {f=-1; continue;}
      else break;
    }
    int N=100;
    cv::cvtColor(frame_in,frame,cv::COLOR_BGR2GRAY);
    frame.copyTo(history[f]);
    if(f>N)  history.erase(f-N-1);
    frame_old= history[((f-ni)>=0?(f-ni):0)];

    // medianBlur(frame, frame, 9);

    cv::Mat velx(frame.rows, frame.cols, CV_32FC1);
    cv::Mat vely(frame.rows, frame.cols, CV_32FC1);
    velx= cv::Scalar(0);
    vely= cv::Scalar(0);

    {
      const cv::Mat &prev(frame_old), &curr(frame);
      std::vector<cv::Point2f> points[2];
      std::vector<uchar> status;
      std::vector<float> err;
// std::cerr<<"p1"<<std::endl;
      cv::goodFeaturesToTrack(prev, points[0], max_feat_count, quality_level, min_distance, cv::Mat(), block_size, use_harris, harris_k);
      cv::cornerSubPix(prev, points[0], cv::Size(subpix_win_size,subpix_win_size), cv::Size(-1,-1), term_criteria);

// std::cerr<<"p2"<<std::endl;
      if(points[0].empty())
      {
      }
      else
      {
        int flag_map[]={0,cv::OPTFLOW_LK_GET_MIN_EIGENVALS};
        int flags= flag_map[optflow_flags];
        cv::calcOpticalFlowPyrLK(prev, curr, points[0], points[1], status, err, cv::Size(lk_win_size,lk_win_size),
                              max_level, term_criteria, flags, min_eig_th);
// std::cerr<<"p3"<<std::endl;
        for(int i(0); i<points[1].size(); ++i)
        {
          if(!status[i])  continue;
          int x=points[1][i].x, y=points[1][i].y;
          if(x>=0 && x<frame.cols && y>=0 && y<frame.rows)
          {
            velx.at<float>(y,x)= x-points[0][i].x;
            vely.at<float>(y,x)= y-points[0][i].y;
          }
          else
          {
// std::cerr<<"p3.5:"<<x<<","<<y<<std::endl;
          }
        }
      }
// std::cerr<<"p4"<<std::endl;
    }

    // visualization
    frame_in*= 0.7;
    {
      cv::Scalar col;
      float vx,vy,spd,angle;
      int dx,dy;
      const float dt(1.0);
      int step(1);
      for (int i(step); i<frame_in.cols-step; i+=step)
      {
        for (int j(step); j<frame_in.rows-step; j+=step)
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

    cv::imshow("camera", frame_in);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
