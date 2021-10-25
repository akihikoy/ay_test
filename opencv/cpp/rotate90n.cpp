// g++ -g -Wall -O2 -o rotate90n.out rotate90n.cpp -lopencv_core -lopencv_highgui -lopencv_videoio
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
#include "rotate90n.h"
using namespace loco_rabbits;

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

  cv::namedWindow("camera",1);
  cv::Mat frame;
  int N(0);
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    Rotate90N(frame,frame,N);
    cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c=='[') {--N; std::cerr<<"N= "<<N<<std::endl;}
    if(c==']') {++N; std::cerr<<"N= "<<N<<std::endl;}
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
