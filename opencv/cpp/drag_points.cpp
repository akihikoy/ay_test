//-------------------------------------------------------------------------------------------
/*! \file    drag_points.cpp
    \brief   Test of dragging points.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.06, 2019

g++ -g -Wall -O2 -o drag_points.out drag_points.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -I/usr/include/opencv4

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

std::vector<cv::Point> Points;
cv::Point SelectOffset;
int SelectedIndex(-1);
const float SELECT_DIST(20.0);

void OnMouse(int event, int x, int y, int flags, void*)
{
  if(event==cv::EVENT_LBUTTONDOWN)
  {
    cv::Point clicked(x,y);
    float min_dist(1.0e10);
    int min_idx(0);
    for(int i(0),i_end(Points.size()); i<i_end; ++i)
    {
      float dist= cv::norm(Points[i]-clicked);
      if(dist<min_dist)
      {
        min_dist= dist;
        min_idx= i;
      }
    }
    if(min_dist<SELECT_DIST)
    {
      SelectedIndex= min_idx;
      SelectOffset= Points[SelectedIndex]-clicked;
    }
    else
      SelectedIndex= -1;
  }
  else if((event==cv::EVENT_MOUSEMOVE && flags&cv::EVENT_FLAG_LBUTTON) && SelectedIndex>=0)
  {
    cv::Point clicked(x,y);
    Points[SelectedIndex]= clicked + SelectOffset;
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);
  cv::setMouseCallback("camera", OnMouse);
  cv::Mat frame;

  cv::RNG rng(0xFFFFFFFF);
  cap>>frame;
  for(int i(0); i<20; ++i)
    Points.push_back(cv::Point(rng.uniform(0,frame.cols),rng.uniform(0,frame.rows)));

  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    for(std::vector<cv::Point>::iterator itr(Points.begin()),itr_end(Points.end()); itr!=itr_end; ++itr)
    {
      if(itr->x<0)  itr->x= 0;
      if(itr->x>frame.cols-1)  itr->x= frame.cols-1;
      if(itr->y<0)  itr->y= 0;
      if(itr->y>frame.rows-1)  itr->y= frame.rows-1;
    }

    for(std::vector<cv::Point>::const_iterator itr(Points.begin()),itr_end(Points.end()); itr!=itr_end; ++itr)
      cv::circle(frame, *itr, 3, cv::Scalar(255,0,255), 2);
    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
