//-------------------------------------------------------------------------------------------
/*! \file    cv2-capture_fps2.cpp
    \brief   Capture with specific FPS by v4l2
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.22, 2016

g++ -g -Wall -O2 -o cv2-capture_fps2.out cv2-capture_fps2.cpp -I$HOME/.local/include -L$HOME/.local/lib -Wl,-rpath=$HOME/.local/lib -lopencv_core -lopencv_highgui -lv4l2

This code failed:
  SetFPS ERROR: V4L: Unable to set camera FPS
    errno= 16  (EBUSY)
  EINTR,EACCES,EBUSY,ENXIO,ENOMEM,EMFILE,ENFILE=4,13,16,6,12,24,23,
Perhaps it is conflicting with the main capture object.
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday

#include <errno.h>
#include <fcntl.h>
// #include <asm/types.h>          /* for videodev2.h */
// #include <sys/ioctl.h>
#include <linux/videodev2.h>
// #include <libv4l1.h>
#include <libv4l2.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

// IOCTL handling for V4L2
#ifdef HAVE_IOCTL_ULONG
static int xioctl( int fd, unsigned long request, void *arg)
#else
static int xioctl( int fd, int request, void *arg)
#endif
{
  int r;
  do r = v4l2_ioctl (fd, request, arg);
  while (-1 == r && EINTR == errno);
  return r;
}

// e.g. SetFPS(cap, 1, 30); // Set 30 FPS
void SetFPS(int device, int numerator, int denominator)
{
  char deviceName[/*MAX_DEVICE_DRIVER_NAME=*/80];
  sprintf(deviceName, "/dev/video%1d", device);
  int deviceHandle = v4l2_open (deviceName, O_RDWR /* required */ | O_NONBLOCK, 0);
  if (deviceHandle < 0)
  {
    fprintf(stderr, "SetFPS ERROR: Unable to open camera %s\n", deviceName);
    fprintf(stderr, "  errno= %d\n", errno);
    v4l2_close(deviceHandle);
  }
  fprintf(stderr, "#SetFPS: deviceHandle= %d\n", deviceHandle);
  fprintf(stderr, "#  errno= %d\n", errno);

  struct v4l2_streamparm setfps;
  memset(&setfps, 0, sizeof(struct v4l2_streamparm));
  setfps.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  setfps.parm.capture.timeperframe.numerator = numerator;
  setfps.parm.capture.timeperframe.denominator = denominator;
  if(xioctl(deviceHandle, VIDIOC_S_PARM, &setfps) < 0)
  {
    fprintf(stderr, "SetFPS ERROR: V4L: Unable to set camera FPS\n");
    fprintf(stderr, "  errno= %d\n", errno);
    std::cerr<<"EINTR,EACCES,EBUSY,ENXIO,ENOMEM,EMFILE,ENFILE="
        <<EINTR<<","<<EACCES<<","<<EBUSY<<","<<ENXIO<<","<<ENOMEM<<","<<EMFILE<<","<<ENFILE<<","<<std::endl;

  }
  v4l2_close(deviceHandle);
}

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
  int cam_id(0);
  cv::VideoCapture cap(cam_id); // open the default camera
  if(argc==2)
  {
    cap.release();
    cam_id= atoi(argv[1]);
    cap.open(cam_id);
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  cap.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
  // cap.set(CV_CAP_PROP_FOURCC,CV_FOURCC('Y','U','Y','V'));
  // cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0);
  cap.set(CV_CAP_PROP_EXPOSURE, 0.0);
  cap.set(CV_CAP_PROP_GAIN, 0.0);

  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
  // cap.set(CV_CAP_PROP_FPS, 15);  // Works with built-in camera of T440p
  // cap.set(CV_CAP_PROP_FPS, 60);
  // cap.set(CV_CAP_PROP_FPS, 120);
  // cap.set(CV_CAP_PROP_FPS, 61612./513.);  // Doesn't work with EPL
  // SetFPS(cam_id, 513, 61612);
  SetFPS(cam_id, 1, 15);

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
