//-------------------------------------------------------------------------------------------
/*! \file    cv2-stereo_fs1.cpp
    \brief   Stereo example with StereoSGBM, StereoRectify
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.01, 2016

g++ -g -Wall -O2 -o cv2-stereo_fs1.out cv2-stereo_fs1.cpp -I$HOME/.local/include -L$HOME/.local/lib -Wl,-rpath=$HOME/.local/lib -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "rotate90n.h"
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
  int cam1(1),cam2(2), n_rotate90(0);
  std::string config_file("data/usbcam4g1_tltr3.yaml");
  if(argc>1)  cam1= atoi(argv[1]);
  if(argc>2)  cam2= atoi(argv[2]);
  if(argc>3)  config_file= argv[3];
  if(argc>4)  n_rotate90= atoi(argv[4]);

  cv::VideoCapture cap1(cam1), cap2(cam2);
  cap1.release();
  cap1.open(cam1);
  cap2.release();
  cap2.open(cam2);
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
  cap1.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
  cap1.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap1.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  cap2.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
  cap2.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap2.set(CV_CAP_PROP_FRAME_HEIGHT, height);

  //From stereo camera calibration:
  // cv::FileStorage fs("data/usbcam4g1_tltr2.yaml", cv::FileStorage::READ);
  cv::FileStorage fs(config_file, cv::FileStorage::READ);
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
  // cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
  // cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
  // Verify with stereoRectify.
  {
    cv::Size img_size2(width/2,height/2);
    cv::Mat vR1, vR2, vP1, vP2, Q;  // output of stereoRectify
    // cv::stereoRectify(K1, D1, K2, D2, img_size, R, T.t(), vR1, vR2, vP1, vP2, Q, /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/0.0 /*, Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0*/);
    // cv::stereoRectify(K1, D1, K2, D2, img_size, R, T.t(), vR1, vR2, vP1, vP2, Q, /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/0.8 /*, Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0*/);
    // D1= D1(cv::Range(0,1),cv::Range(0,4));
    // D2= D2(cv::Range(0,1),cv::Range(0,4));
    // D1= (cv::Mat_<double>(1,4)<<3.3875402099004873e-02, -2.2827110625951449e-01, 3.0986831555716177e-01, -1.3603666141693707e-01);
    // D2= (cv::Mat_<double>(1,4)<<3.3875402099004873e-02, -2.2827110625951449e-01, 3.0986831555716177e-01, -1.3603666141693707e-01);
    cv::fisheye::stereoRectify(K1, D1, K2, D2, img_size, R, T.t(), vR1, vR2, vP1, vP2, Q, /*flags=*/cv::CALIB_ZERO_DISPARITY, img_size2, /*balance=*/0.0, /*fov_scale=*/0.55);
    // print(R1-vR1);
    // print(R2-vR2);
    // print(P1-vP1);
    // print(P2-vP2);
    print(Q);
    R1= vR1;
    R2= vR2;
    P1= vP1;
    P2= vP2;
    cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, img_size2, CV_16SC2, map11, map12);
    cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, img_size2, CV_16SC2, map21, map22);
  }

  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  cv::namedWindow("disparity",1);
  cv::Mat frame1, frame2, disparity;
  cv::Mat gray1, gray2;
  cv::Mat frame1r, frame2r;
  cv::Mat buf;
  int n_disp(16*3), w_size(5), min_disp(0);
  n_disp= (((img_size.width/8) + 15) & -16);
  min_disp= std::max(0,n_disp-16*8);
  for(;;)
  {
    bool stereo_reset(false);
    cv::StereoSGBM stereo(/*minDisparity=*/min_disp, /*numDisparities=*/n_disp, /*SADWindowSize=*/w_size,
                // /*int P1=*/0, /*int P2=*/0, /*int disp12MaxDiff=*/0,
                /*int P1=*/8*3*w_size*w_size, /*int P2=*/32*3*w_size*w_size, /*int disp12MaxDiff=*/0,
                /*int preFilterCap=*/0, /*int uniquenessRatio=*/0,
                /*int speckleWindowSize=*/0, /*int speckleRange=*/0,
                /*bool fullDP=*/false);
    // cv::StereoBM stereo(cv::StereoBM::BASIC_PRESET, /*ndisparities=*/n_disp, /*SADWindowSize=*/w_size);
    for(;!stereo_reset;)
    {
      cap1 >> frame1;
      cap2 >> frame2;
      Rotate90N(frame1,frame1,n_rotate90);
      Rotate90N(frame2,frame2,n_rotate90);

      cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
      cv::cvtColor(frame2, gray2, CV_BGR2GRAY);
      frame1= gray1;
      frame2= gray2;

      cv::remap(frame1, frame1r, map11, map12, cv::INTER_LINEAR);
      cv::remap(frame2, frame2r, map21, map22, cv::INTER_LINEAR);
      frame1= frame1r;
      frame2= frame2r;
      // TODO:FIXME: REDUCE THE IMAGE SIZES

      // StereoSGBM:
      stereo(frame1, frame2, disparity);
      // StereoSGBM (gray):
      // cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
      // cv::cvtColor(frame2, gray2, CV_BGR2GRAY);
      // stereo(gray1, gray2, disparity);
      // StereoBM:
      // cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
      // cv::cvtColor(frame2, gray2, CV_BGR2GRAY);
      // stereo(gray1, gray2, disparity, /*disptype=*/CV_16S);

      // cv::filterSpeckles(disparity, /*newVal=*/0, /*maxSpeckleSize=*/10, /*maxDiff=*/16, buf);

      cv::normalize(disparity, disparity, 0, 255, CV_MINMAX, CV_8U);
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
