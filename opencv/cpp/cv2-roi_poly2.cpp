//-------------------------------------------------------------------------------------------
/*! \file    cv2-roi_poly2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2016

g++ -g -Wall -O2 -o cv2-roi_poly2.out cv2-roi_poly2.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
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
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

  cv::RNG rng(cv::getTickCount());
  cv::Mat points(6,2,CV_32S), hull;
  cv::namedWindow("camera",1);
  cv::namedWindow("mask",1);
  cv::Mat frame, mask, masked;
  for(int f(0);;++f)
  {
    cap >> frame; // get a new frame from camera

    if(f%10==0)
      for(int i(0);i<6;++i)
      {
        points.at<int>(i,0)= rng.uniform(0,320);
        points.at<int>(i,1)= rng.uniform(0,240);
      }

    cv::convexHull(points, hull/*, bool clockwise=false, bool returnPoints=true */);
    std::vector<cv::Mat> ppt(1);
    ppt[0]= hull;

    mask.create(frame.size(), CV_8U);
    mask.setTo(0);
    cv::fillPoly(mask, ppt, cv::Scalar(255));

    masked.setTo(0);
    frame.copyTo(masked, mask);

    cv::imshow("camera", masked);
    cv::imshow("mask", mask);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
