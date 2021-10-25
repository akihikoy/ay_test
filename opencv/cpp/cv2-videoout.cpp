// g++ -g -Wall -O2 -o cv2-videoout.out cv2-videoout.cpp -lopencv_core -lopencv_highgui -lopencv_videoio

// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <string>
#include <cstdio>
#include <sys/time.h>  // gettimeofday

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}

bool OpenVideoOut(cv::VideoWriter &vout, const char *file_name, int fps, const cv::Size &size)
{
  // int codec= CV_FOURCC('P','I','M','1');  // mpeg1video
  // int codec= CV_FOURCC('X','2','6','4');  // x264?
  int codec= CV_FOURCC('m','p','4','v');  // mpeg4 (Simple Profile)
  vout.open(file_name, codec, fps, size, true);

  if (!vout.isOpened())
  {
    std::cout<<"Failed to open the output video: "<<file_name<<std::endl;
    return false;
  }
  return true;
}

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

  cv::VideoWriter vout;
  double time_prev= GetCurrentTime(), fps(10.0), fps_alpha(0.1);

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    cv::Mat neg_img= ~frame;
    if(vout.isOpened())  vout<<neg_img;

    // cv::imshow("camera", frame);
    cv::imshow("camera", neg_img);


    fps= fps_alpha*(1.0/(GetCurrentTime()-time_prev)) + (1.0-fps_alpha)*fps;
    // fps= 50.0;
    time_prev= GetCurrentTime();
    std::cout<<"fps: "<<fps<<std::endl;

    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c=='w')
    {
      if(vout.isOpened())  vout.release();
      else  OpenVideoOut(vout, "/tmp/test-cv2-video.avi", fps, cv::Size(frame.cols,frame.rows));
    }
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
