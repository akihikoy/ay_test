//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-sobel-lk.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.08, 2017

g++ -I -Wall -O2 optical-flow-sobel-lk.cpp -o optical-flow-sobel-lk.out -lopencv_core -lopencv_legacy -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <map>
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

  const char *window("Optical Flow Sobel+LK");
  cv::namedWindow(window,1);
  int ni(4);
  cv::createTrackbar( "Interval:", window, &ni, 100, NULL);

  cv::Mat frame_in, frame, frame_old;
  std::map<int,cv::Mat> history;
  for(int i(0);;++i)
  {
    int N= 100;
    cap >> frame_in;
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);

    cv::blur(frame, frame, cv::Size(3,3));

    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
    cv::Sobel(frame, grad_x, CV_16S, 1, 0, 3, /*scale*/1, /*delta*/0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Sobel(frame, grad_y, CV_16S, 0, 1, 3, /*scale*/1, /*delta*/0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    frame= 0.5*abs_grad_x+0.5*abs_grad_y;

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
    frame_in*= 0.3;
    cv::Mat edges[3]= {frame, frame, frame}, edge;
    cv::merge(edges,3,edge);
    frame_in+= 0.5*edge;
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
          if(spd<4.0 || 1000.0<spd)  continue;
          angle= std::atan2(vy,vx);
          col= CV_RGB(0.0,255.0*std::fabs(std::cos(angle)),255.0*std::fabs(std::sin(angle)));
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
