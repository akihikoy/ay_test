// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>

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
  for(int i(0); i<100; ++i)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);

    std::stringstream file_name;
    file_name<<"frame/frame"<<std::setfill('0')<<std::setw(4)<<i<<".jpg";
    cv::imwrite(file_name.str(), frame);
    std::cout<<"Saved "<<file_name.str()<<std::endl;

    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
