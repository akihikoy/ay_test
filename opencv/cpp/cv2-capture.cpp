// g++ -g -Wall -O2 -o cv2-capture.out cv2-capture.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
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

int main(int argc, char **argv)
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

  // cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  // cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('Y','U','Y','V'));
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
  // cap.set(cv::CAP_PROP_EXPOSURE, 0.0);
  // cap.set(cv::CAP_PROP_GAIN, 0.0);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 800);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  // cap.set(cv::CAP_PROP_FPS, 10);

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
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
