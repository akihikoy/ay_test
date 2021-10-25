//-------------------------------------------------------------------------------------------
/*! \file    cv2-capture2.cpp
    \brief   Capture from two cameras
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.07, 2016

g++ -g -Wall -O2 -o cv2-capture2.out cv2-capture2.cpp -lopencv_core -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
  // return ros::Time::now().toSec();
}
//-------------------------------------------------------------------------------------------

struct TFPSEstimator
{
  double Alpha;
  double FPS;
  double TimePrev;
  TFPSEstimator(const double &init_fps=10.0, const double &alpha=0.05);
  void Step();
};
//-------------------------------------------------------------------------------------------
TFPSEstimator::TFPSEstimator(const double &init_fps, const double &alpha)
  :
    Alpha (alpha),
    FPS (init_fps),
    TimePrev (-1.0)
{
}
void TFPSEstimator::Step()
{
  if(TimePrev<0.0)
  {
    TimePrev= GetCurrentTime();
  }
  else
  {
    double new_fps= 1.0/(GetCurrentTime()-TimePrev);
    if(new_fps>FPS/20.0 && new_fps<FPS*20.0)  // Removing outliers (e.g. pause/resume)
      FPS= Alpha*new_fps + (1.0-Alpha)*FPS;
    TimePrev= GetCurrentTime();
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap1(1), cap2(2);
  if(argc==3)
  {
    cap1.release();
    cap1.open(atoi(argv[1]));
    cap2.release();
    cap2.open(atoi(argv[2]));
  }
  if(!cap1.isOpened() || !cap2.isOpened())
  {
    std::cerr<<"failed to open camera(s)!"<<std::endl;
    return -1;
  }
  std::cerr<<"cameras opened"<<std::endl;

  // set resolution
  cap1.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
  // cap1.set(CV_CAP_PROP_FOURCC,CV_FOURCC('H','2','6','4'));
  // cap1.set(CV_CAP_PROP_FOURCC,CV_FOURCC('Y','U','Y','V'));
  cap1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  // cap1.set(CV_CAP_PROP_FRAME_WIDTH, 800);
  // cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
  // cap1.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
  // cap1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  // cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

  cap2.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
  // cap2.set(CV_CAP_PROP_FOURCC,CV_FOURCC('H','2','6','4'));
  // cap2.set(CV_CAP_PROP_FOURCC,CV_FOURCC('Y','U','Y','V'));
  cap2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  // cap1.set(CV_CAP_PROP_FRAME_WIDTH, 800);
  // cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
  // cap2.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
  // cap2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  // cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

  TFPSEstimator fps;
  int show_fps(0);
  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  cv::Mat frame1,frame2;
  for(;;)
  {
    cap1 >> frame1;
    cap2 >> frame2;
    cv::imshow("camera1", frame1);
    cv::imshow("camera2", frame2);
    fps.Step();
    if(show_fps==0)
    {
      std::cerr<<"FPS: "<<fps.FPS<<std::endl;
      show_fps= fps.FPS*4;
    }
    --show_fps;
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
