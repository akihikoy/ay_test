//-------------------------------------------------------------------------------------------
/*! \file    cv2-filter2d.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.08, 2016

g++ -g -Wall -O2 -o cv2-filter2d.out cv2-filter2d.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
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
  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

  cv::Mat kernel(cv::Size(1,100),CV_32F);
  kernel= cv::Mat::ones(kernel.size(),CV_32F)/(float)(kernel.rows*kernel.cols);

  cv::namedWindow("camera",1);
  cv::Mat frame, filtered;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    cv::filter2D(frame, filtered, /*ddepth=*/-1, kernel);

    cv::imshow("camera", filtered);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
