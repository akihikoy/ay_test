//-------------------------------------------------------------------------------------------
/*! \file    cv2-roi_poly.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.29, 2016

g++ -g -Wall -O2 -o cv2-roi_poly.out cv2-roi_poly.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
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
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

  cv::namedWindow("camera",1);
  cv::namedWindow("mask",1);
  cv::Mat frame, mask, masked;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    // cv::Point points[1][4]= {{cv::Point(128,30), cv::Point(180,120), cv::Point(150,140), cv::Point(100,100)}};
    // const cv::Point *ppt[1]= {points[0]};
    // int n_points[]= {4};

    // std::vector<cv::Point> points;
    // points.reserve(4);
    // points.push_back(cv::Point(128,30));
    // points.push_back(cv::Point(180,120));
    // points.push_back(cv::Point(150,140));
    // points.push_back(cv::Point(100,100));
    // const cv::Point *ppt[1] = {&points[0]};
    // int n_points[]= {(int)points.size()};

    // mask.create(frame.size(), CV_8U);
    // mask.setTo(0);
    // cv::fillPoly(mask, ppt, n_points, 1, cv::Scalar(255));

    cv::Mat points(4,2,CV_32S);
    points= (cv::Mat_<int>(4,2)<<
          128,30 ,
          180,120,
          150,140,
          100,100);
    std::vector<cv::Mat> ppt(1);
    ppt[0]= points;

    mask.create(frame.size(), CV_8U);
    mask.setTo(0);
    cv::fillPoly(mask, ppt, cv::Scalar(255));

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
