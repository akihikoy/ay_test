//-------------------------------------------------------------------------------------------
/*! \file    cv2-cap-stream.cpp
    \brief   Capture from stream
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.07, 2016

g++ -g -Wall -O2 -o cv2-cap-stream.out cv2-cap-stream.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
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
  cv::VideoCapture cap1;
  cap1.open("http://localhost:8080/?action=stream?dummy=file.mjpg");
  if(argc==2)
  {
    cap1.release();
    cap1.open(argv[1]);
  }
  if(!cap1.isOpened())
  {
    std::cerr<<"failed to open camera(s)!"<<std::endl;
    return -1;
  }
  std::cerr<<"cameras opened"<<std::endl;

  // set resolution
  // cap1.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  // cap1.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('H','2','6','4'));
  // cap1.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('Y','U','Y','V'));
  // cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  // cap1.set(cv::CAP_PROP_FRAME_WIDTH, 800);
  // cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
  // cap1.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  // cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

  TFPSEstimator fps;
  int show_fps(0);
  cv::namedWindow("camera1",1);
  cv::Mat frame1;
  for(;;)
  {
    cap1 >> frame1;
    cv::imshow("camera1", frame1);
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
