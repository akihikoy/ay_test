//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-lk2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016

g++ -I -Wall -O2 optical-flow-lk2.cpp -o optical-flow-lk2.out -lopencv_core -lopencv_legacy -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "cap_open.h"
#include "cv2-videoout2.h"
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

  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
  // cap.set(CV_CAP_PROP_FPS, 60);  // Works with ELP USBFHD01M-L180

  TEasyVideoOut vout[2];
  vout[0].SetfilePrefix("/tmp/optflow_in");
  vout[1].SetfilePrefix("/tmp/optflow_fl");
  int show_fps(0);

  cv::namedWindow("camera",1);
  cv::namedWindow("optical flow",1);
  cv::Mat frame_in, frame, frame_old, frame_disp;
  cap >> frame;
  cv::resize(frame,frame,cv::Size(640,480),CV_INTER_LINEAR);
  cv::cvtColor(frame,frame,CV_BGR2GRAY);
  for(int i(0);;++i)
  {
    frame.copyTo(frame_old);
    if(!cap.Read(frame_in))
    {
      if(cap.WaitReopen()) {i=-1; continue;}
      else break;
    }
    cv::resize(frame_in,frame_in,cv::Size(640,480),CV_INTER_LINEAR);
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);

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
    // frame_in*= 0.5;
    frame_disp.create(frame_in.size(), frame_in.type());
    frame_disp.setTo(0);
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
          spd= /*std::sqrt*/(vx*vx+vy*vy);
          if(spd<3.0*3.0 || 1000.0*1000.0<spd)  continue;
          angle= std::atan2(vy,vx);
          // col= CV_RGB(0.0,255.0*std::fabs(std::cos(angle)),255.0*std::fabs(std::sin(angle)));
          col= CV_RGB(0.0,255.0,255.0);
          // cv::line(frame_in, cv::Point(i,j), cv::Point(i,j)+cv::Point(dt*vx,dt*vy), col, 1);
          // cv::circle(frame_in, cv::Point(i,j), 1, col);
          frame_disp.at<cv::Vec3b>(j,i)= cv::Vec3b(col[0],col[1],col[2]);
        }
      }
    }
    cv::erode(frame_disp,frame_disp,cv::Mat(),cv::Point(-1,-1), 1);
    cv::dilate(frame_disp,frame_disp,cv::Mat(),cv::Point(-1,-1), 1);

    vout[0].Step(frame_in);
    vout[0].VizRec(frame_in);
    vout[1].Step(frame_disp);
    vout[1].VizRec(frame_disp);

    cv::imshow("camera", frame_in);
    cv::imshow("optical flow", frame_disp);

    if(show_fps==0)
    {
      std::cerr<<"FPS: "<<vout[0].FPS()<<", "<<vout[1].FPS()<<std::endl;
      show_fps=vout[0].FPS()*4;
    }
    --show_fps;

    int c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    else if(char(c)=='W')
    {
      vout[0].Switch();
      vout[1].Switch();
    }
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
