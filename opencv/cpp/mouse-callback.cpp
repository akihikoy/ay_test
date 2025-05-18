// g++ -O2 mouse-callback.cpp -o mouse-callback.out -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
// NOTE: Click a point on the window, then the BGR and HSV color of the point is displayed.

// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>  // cvtColor
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <cstdio>

void OnMouse(int event, int x, int y, int flags, void *vpimg)
{
  std::cerr<<"event= "<<event<<" flags="<<flags<<std::endl;
  std::cerr<<cv::EVENT_LBUTTONDOWN<<"/"<<cv::EVENT_RBUTTONDOWN;
  std::cerr<<"//"<<cv::EVENT_FLAG_SHIFTKEY<<std::endl;
  if(event != cv::EVENT_LBUTTONDOWN)
    return;

  cv::Mat *pimg(reinterpret_cast<cv::Mat*>(vpimg));
  cv::Mat original(1,1,pimg->type()), converted;
  original.at<cv::Vec3b>(0,0)= pimg->at<cv::Vec3b>(y,x);  /* WARNING: be careful about the order of y and x WARNING */
  cv::cvtColor(original, converted, cv::COLOR_BGR2HSV);
  std::cout<< "BGR: "<<original.at<cv::Vec3b>(0,0)<<"  HSV: "<<converted.at<cv::Vec3b>(0,0)<<std::endl;
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

  cv::namedWindow("camera",1);
  cv::Mat frame;
  cv::setMouseCallback("camera", OnMouse, &frame);
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
