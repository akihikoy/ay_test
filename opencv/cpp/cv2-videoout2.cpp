//-------------------------------------------------------------------------------------------
/*! \file    cv2-videoout2.cpp
    \brief   Easy video output tool.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.17, 2015

    g++ -g -Wall -O2 -o cv2-videoout2.out cv2-videoout2.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "cv2-videoout2.h"
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

  TEasyVideoOut vout;
  vout.SetfilePrefix("/tmp/vout");

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    cv::Mat neg_img= ~frame;

    vout.Step(neg_img);
    vout.VizRec(neg_img);
    cv::imshow("camera", neg_img);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c=='w')  vout.Switch();
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
