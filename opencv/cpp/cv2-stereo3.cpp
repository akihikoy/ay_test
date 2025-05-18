//-------------------------------------------------------------------------------------------
/*! \file    cv2-stereo3.cpp
    \brief   Stereo example with StereoSGBM, StereoRectify
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.01, 2016

g++ -g -Wall -O2 -o cv2-stereo3.out cv2-stereo3.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
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
  cap1.release();
  cap1.open(1);
  cap2.release();
  cap2.open(2);
  if(!cap1.isOpened() || !cap2.isOpened())
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  int width(640), height(480);
  // int width(480), height(420);
  // int width(352), height(288);
  // int width(320), height(240);
  cap1.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  cap1.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap1.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  cap2.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  cap2.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap2.set(cv::CAP_PROP_FRAME_HEIGHT, height);

  //From stereo camera calibration:
  // cv::FileStorage fs("data/ext_usbcam1_stereo3.yaml", cv::FileStorage::READ);
  // cv::FileStorage fs("data/usbcam4g1_tltr1.yaml", cv::FileStorage::READ);
  cv::FileStorage fs("data/usbcam2f1.yaml", cv::FileStorage::READ);
  if(!fs.isOpened())
  {
    std::cerr<<"Failed to open file"<<std::endl;
    return -1;
  }
  cv::Mat D1, K1, D2, K2;  // distortion, intrinsics
  cv::Mat R1, R2, P1, P2;  // output of stereoRectify
  cv::Mat R, T;  // rotation, translation between cameras
  fs["D1"] >> D1;
  fs["K1"] >> K1;
  fs["P1"] >> P1;
  fs["R1"] >> R1;
  fs["D2"] >> D2;
  fs["K2"] >> K2;
  fs["P2"] >> P2;
  fs["R2"] >> R2;
  fs["R"] >> R;
  fs["T"] >> T;
  cv::Mat map11, map12, map21, map22;
  cv::Size img_size(width,height);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
  // Verify with stereoRectify.
  {
    cv::Mat vR1, vR2, vP1, vP2, Q;  // output of stereoRectify
    cv::stereoRectify(K1, D1, K2, D2, img_size, R, T.t(), vR1, vR2, vP1, vP2, Q, /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/0.0 /*, Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0*/);
    print(R1-vR1);
    print(R2-vR2);
    print(P1-vP1);
    print(P2-vP2);
    print(Q);
    // R1= vR1;
    // R2= vR2;
    // P1= vP1;
    // P2= vP2;
    // cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    // cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
  }

  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  cv::namedWindow("disparity",1);
  cv::Mat frame1, frame2, disparity;
  cv::Mat gray1, gray2;
  cv::Mat frame1r, frame2r;
  cv::Mat buf;
  int n_disp(16*3), w_size(5), min_disp(0);
  n_disp= (((img_size.width/8) + 15) & -16)+16*4;
  min_disp= std::max(0,n_disp-16*8);
  for(;;)
  {
    bool stereo_reset(false);
    cv::Ptr<cv::StereoSGBM> stereo=cv::StereoSGBM::create(/*minDisparity=*/min_disp, /*numDisparities=*/n_disp, /*blockSize=*/w_size,
                // /*int P1=*/0, /*int P2=*/0, /*int disp12MaxDiff=*/0,
                /*int P1=*/8*3*w_size*w_size, /*int P2=*/32*3*w_size*w_size, /*int disp12MaxDiff=*/0,
                /*int preFilterCap=*/0, /*int uniquenessRatio=*/0,
                /*int speckleWindowSize=*/0, /*int speckleRange=*/0
                /*, mode*/);
    // cv::Ptr<cv::StereoBM> stereo=cv::StereoBM::create(/*ndisparities=*/n_disp, /*blockSize=*/w_size);
    for(;!stereo_reset;)
    {
      cap1 >> frame1;
      cap2 >> frame2;

      cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
      cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
      frame1= gray1;
      frame2= gray2;

      cv::remap(frame1, frame1r, map11, map12, cv::INTER_LINEAR);
      cv::remap(frame2, frame2r, map21, map22, cv::INTER_LINEAR);
      frame1= frame1r;
      frame2= frame2r;
      // TODO:FIXME: REDUCE THE IMAGE SIZES

      // StereoSGBM:
      stereo->compute(frame1, frame2, disparity);
      // StereoSGBM (gray):
      // cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
      // cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
      // stereo->compute(gray1, gray2, disparity);
      // StereoBM:
      // cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
      // cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
      // stereo->compute(gray1, gray2, disparity);

      // cv::filterSpeckles(disparity, /*newVal=*/0, /*maxSpeckleSize=*/10, /*maxDiff=*/16, buf);

      cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::imshow("camera1", frame1);
      cv::imshow("camera2", frame2);
      cv::imshow("disparity", disparity);
      // std::cerr<<disparity<<std::endl;
      int c(cv::waitKey(10));
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
