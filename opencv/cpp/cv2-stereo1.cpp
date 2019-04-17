//-------------------------------------------------------------------------------------------
/*! \file    cv2-stereo1.cpp
    \brief   Stereo example with StereoBM
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Mar.29, 2016

g++ -g -Wall -O2 -o cv2-stereo1.out cv2-stereo1.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap1(1), cap2(2);
  if(!cap1.isOpened() || !cap2.isOpened())
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  // int width(640), height(480);
  // int width(480), height(420);
  // int width(320), height(240);
  int width(640), height(360);
  cap1.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap1.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  cap2.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap2.set(CV_CAP_PROP_FRAME_HEIGHT, height);

  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  cv::namedWindow("disparity",1);
  cv::Mat frame1, frame2, gray1, gray2, disparity;
  int n_disp(16*3), w_size(5);
  cv::StereoBM stereo(cv::StereoBM::BASIC_PRESET, /*ndisparities=*/n_disp, /*SADWindowSize=*/w_size);
  for(;;)
  {
    cap1 >> frame1;
    cap2 >> frame2;
    cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
    cv::cvtColor(frame2, gray2, CV_BGR2GRAY);
    // stereo(gray1, gray2, disparity, /*disptype=*/CV_16S);
    // disparity+= 16;
    stereo(gray1, gray2, disparity, /*disptype=*/CV_32FC1);
    disparity+= 1.0;
    disparity/=100.0;
    cv::imshow("camera1", frame1);
    cv::imshow("camera2", frame2);
    cv::imshow("disparity", disparity);
    // std::cerr<<disparity<<std::endl;
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c=='[' or c==']' or c==',' or c=='.')
    {
      if     (c=='[')  {n_disp-= 16; if(n_disp<0) n_disp=0; print(n_disp);}
      else if(c==']')  {n_disp+= 16; print(n_disp);}
      else if(c==',')  {w_size-= 2; if(w_size<=5) w_size=5; print(w_size);}
      else if(c=='.')  {w_size+= 2; print(w_size);}
      stereo.init(cv::StereoBM::BASIC_PRESET, /*ndisparities=*/n_disp, /*SADWindowSize=*/w_size);
    }
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
