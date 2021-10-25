//-------------------------------------------------------------------------------------------
/*! \file    mouse_poyly.cpp
    \brief   Draw polygon by clicking.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.21, 2018

g++ -g -Wall -O2 -o mouse_poyly.out mouse_poyly.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc
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

void OnMouse(int event, int x, int y, int flags, void *vp_polygon)
{
  std::vector<std::vector<cv::Point> >  &polygon(*reinterpret_cast<std::vector<std::vector<cv::Point> >*>(vp_polygon));

  if(event==cv::EVENT_LBUTTONUP)
  {
    polygon[0].push_back(cv::Point(x,y));
  }
  else if(event==cv::EVENT_RBUTTONUP)
  {
    for(size_t i(0); i<polygon[0].size(); ++i)
      std::cout<<" "<<polygon[0][i];
    std::cout<<std::endl;
    polygon[0].clear();
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  std::vector<std::vector<cv::Point> >  polygon(1);

  cv::namedWindow("camera",1);
  cv::setMouseCallback("camera", OnMouse, &polygon);
  cv::Mat frame;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    if(polygon[0].size()>0)
    {
      cv::fillPoly(frame, polygon, CV_RGB(128,0,128));
      cv::polylines(frame, polygon, /*isClosed=*/true, CV_RGB(255,0,255), 2);
    }

    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
