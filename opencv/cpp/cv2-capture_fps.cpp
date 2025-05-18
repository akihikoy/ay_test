//-------------------------------------------------------------------------------------------
/*! \file    cv2-capture_fps.cpp
    \brief   Capture with specific FPS
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016
    \version 0.2
    \date    Mar.16, 2022

g++ -g -Wall -O2 -o cv2-capture_fps.out cv2-capture_fps.cpp -lopencv_core -lopencv_highgui  -lopencv_videoio -I/usr/include/opencv4
g++ -g -Wall -O2 -o cv2-capture_fps.out cv2-capture_fps.cpp -I$HOME/.local/include -L$HOME/.local/lib -Wl,-rpath=$HOME/.local/lib -lopencv_core -lopencv_highgui  -lopencv_videoio -I/usr/include/opencv4

xxx We have to use source-build OpenCV otherwise we cannot set cv::CAP_PROP_FPS:
xxx   HIGHGUI ERROR: V4L: Property <unknown property string>(5) not supported by device
Note@2022-03-16 with repository-installed OpenCV 3.2.0: cv::CAP_PROP_FPS is available.
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

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
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  // cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('Y','U','Y','V'));
  // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
  // cap.set(cv::CAP_PROP_EXPOSURE, 0.0);
  // cap.set(cv::CAP_PROP_GAIN, 0.0);

  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  // cap.set(cv::CAP_PROP_FPS, 15);  // Works with built-in camera of T440p
  // cap.set(cv::CAP_PROP_FPS, 60);
  // cap.set(cv::CAP_PROP_FPS, 120);
  // cap.set(cv::CAP_PROP_FPS, 61612./513.);  // Doesn't work with ELP USBFHD01M-L180 as we are using YUYV? BGR3?
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, 60);  // Doesn't work with ELP USBFHD01M-L180 as we are using YUYV? BGR3?
  // Note: cv::CAP_PROP_FPS worked with an Asahi-CM camera.

  TFPSEstimator fps;
  int show_fps(0);
  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);
    fps.Step();
    if(show_fps==0)
    {
      std::cerr<<"FPS: "<<fps.FPS<<std::endl;
      show_fps= fps.FPS*4;
    }
    --show_fps;
    int c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
