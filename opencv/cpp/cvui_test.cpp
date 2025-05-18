//-------------------------------------------------------------------------------------------
/*! \file    cvui_test.cpp
    \brief   Test to use CVUI.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.26, 2018

g++ -g -Wall -O2 -std=c++11 -o cvui_test.out cvui_test.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -I/usr/include/opencv4

Document:
https://www.learnopencv.com/cvui-gui-lib-built-on-top-of-opencv-drawing-primitives/
Src:
https://github.com/spmallick/learnopencv/tree/master/UI-cvui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
#include "cvui.h"
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

void OnMouse(int event, int x, int y, int flags, void *data)
{
  if(event==cv::EVENT_LBUTTONUP)
  {
    std::cout<<x<<" "<<y<<std::endl;
  }
}
//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);
  cvui::init("camera");
  // cv::setMouseCallback("camera", OnMouse);  // WARNING: We cannot define our mouse callback.

  bool flag(true), flag2(false);
  double step(1.0);

  cv::Mat frame;
  for(double count(0.0);;count+=(flag?1.0:-1.0)*step)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    cvui::window(frame, 10, 50, 180, 180, "Window");
    if(cvui::button(frame, 150, 80, "Reset!"))
      count= 0;  // When the button was clicked, reset count.
    cvui::printf(frame, 250, 90, 0.4, 0xff0000, "count: %f", count);
    cvui::checkbox(frame, 15, 80, "Increment", &flag);
    cvui::trackbar(frame, 15, 140, 160, &step, 0.0, 10.0);

    if(flag2)
    {
      cvui::window(frame, 400, 50, 180, 180, "");
      cvui::checkbox(frame, 400, 50, "Menu", &flag2);
      if(cvui::button(frame, 450, 120, "Reset!"))
        count= 0;  // When the button was clicked, reset count.
      cvui::checkbox(frame, 405, 80, "Increment", &flag);
      cvui::trackbar(frame, 405, 140, 160, &step, 0.0, 10.0);
    }
    else
      cvui::checkbox(frame, 400, 50, "Menu", &flag2);

    cvui::update();
    cv::imshow("camera", frame);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;

    // usleep(500*1000);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
