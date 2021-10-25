//-------------------------------------------------------------------------------------------
/*! \file    cv2-stereo2.cpp
    \brief   Stereo example with StereoSGBM
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Mar.29, 2016

g++ -g -Wall -O2 -o cv2-stereo2.out cv2-stereo2.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
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
  cv::Mat frame1, frame2, disparity;
  int n_disp(16*3), w_size(5);
  for(;;)
  {
    bool stereo_reset(false);
    cv::Ptr<cv::StereoSGBM> stereo=cv::StereoSGBM::create(/*minDisparity=*/1, /*numDisparities=*/n_disp, /*blockSize=*/w_size,
                // /*int P1=*/0, /*int P2=*/0, /*int disp12MaxDiff=*/0,
                /*int P1=*/8*3*w_size*w_size, /*int P2=*/32*3*w_size*w_size, /*int disp12MaxDiff=*/0,
                /*int preFilterCap=*/0, /*int uniquenessRatio=*/0,
                /*int speckleWindowSize=*/0, /*int speckleRange=*/0
                /*, mode*/);
    for(;!stereo_reset;)
    {
      cap1 >> frame1;
      cap2 >> frame2;
      stereo->compute(frame1, frame2, disparity);
      cv::normalize(disparity, disparity, 0, 255, CV_MINMAX, CV_8U);
      cv::imshow("camera1", frame1);
      cv::imshow("camera2", frame2);
      cv::imshow("disparity", disparity);
      // std::cerr<<disparity<<std::endl;
      char c(cv::waitKey(10));
      if(c=='\x1b'||c=='q') break;
      else if(c=='[' or c==']' or c==',' or c=='.')
      {
        if     (c=='[')  {n_disp-= 16; if(n_disp<=0) n_disp=16; print(n_disp);}
        else if(c==']')  {n_disp+= 16; print(n_disp);}
        else if(c==',')  {w_size-= 2; if(w_size<=1) w_size=1; print(w_size);}
        else if(c=='.')  {w_size+= 2; print(w_size);}
        stereo_reset= true;
      }
      // usleep(10000);
    }
    if(!stereo_reset)  break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
