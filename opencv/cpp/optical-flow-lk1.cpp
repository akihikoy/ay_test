//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-lk1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016

g++ -I -Wall -O2 optical-flow-lk1.cpp -o optical-flow-lk1.out -lopencv_core -lopencv_legacy -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <map>
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

  const char *window("Optical Flow LK");
  cv::namedWindow(window,1);
  int ni(10);
  float v_min(4.0), v_max(1000.0);
  CreateTrackbar<int>("Interval", window, &ni, 0, 100, 1,  NULL);
  CreateTrackbar<float>("v_min", window, &v_min, 0.0f, 100.0f, 0.1f,  NULL);
  CreateTrackbar<float>("v_max", window, &v_max, 0.0f, 1000.0f, 0.01f,  NULL);

  cv::Mat frame_in, frame, frame_old;
  std::map<int,cv::Mat> history;
  for(int i(0);;++i)
  {
    if(!cap.Read(frame_in))
    {
      if(cap.WaitReopen()) {i=-1; continue;}
      else break;
    }
    int N=100;
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);
    frame.copyTo(history[i]);
    if(i>N)  history.erase(i-N-1);
    frame_old= history[((i-ni)>=0?(i-ni):0)];

    // medianBlur(frame, frame, 9);

    cv::Mat velx(frame.rows, frame.cols, CV_32FC1);
    cv::Mat vely(frame.rows, frame.cols, CV_32FC1);
    velx= cv::Scalar(0);
    vely= cv::Scalar(0);
    CvMat prev(frame_old), curr(frame), velx2(velx), vely2(vely);
    // Using HS:
    // CvTermCriteria criteria= cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 16, 0.1);
    // cvCalcOpticalFlowHS(&prev, &curr, 0, &velx2, &vely2, 10.0, criteria);
    // Using LK:
    cvCalcOpticalFlowLK(&prev, &curr, cv::Size(5,5), &velx2, &vely2);

    // visualization
    frame_in*= 0.7;
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
