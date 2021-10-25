//-------------------------------------------------------------------------------------------
/*! \file    cv2-stereo_cmp.cpp
    \brief   Comparing stereo methods.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.15, 2021
g++ -g -Wall -O2 -o cv2-stereo_cmp.out cv2-stereo_cmp.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

e.g.
$ ./cv2-stereo_cmp.out sample/tsukuba_l.png sample/tsukuba_r.png
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

// Project a depth image I(x,y)=z to I(x,z)=y.
cv::Mat ProjectDepth(const cv::Mat &depth, int d_max=255)
{
  cv::Mat out(cv::Size(depth.cols,d_max),CV_8UC3);
  out.setTo(0);
  for(int y(0); y<depth.rows; ++y)
  {
    float r= float(y)/float(depth.rows);
    cv::Vec3b col= cv::Vec3b((1.0-r)*255.0,128,r*255.0);
    for(int x(0); x<depth.cols; ++x)
    {
      int z= d_max-1-std::max(0,std::min(d_max-1, int(depth.at<unsigned char>(y,x)) ));
      out.at<cv::Vec3b>(z,x)= col;
    }
  }
  return out;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat frame1= cv::imread(argv[1]);
  cv::Mat frame2= cv::imread(argv[2]);

  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  cv::namedWindow("disparity",1);
  cv::namedWindow("projected",1);

  int method(0);  // 0: StereoBM, 1: StereoSGBM
  int n_disp(16*3), w_size(5);

  CreateTrackbar<int>("method", "disparity", &method, 0, 1, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_disp", "disparity", &n_disp, 16, 16*20, 16,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("w_size", "disparity", &w_size, 5, 255, 2,  &TrackbarPrintOnTrack);

  cv::Mat gray1, gray2, disparity;

  for(;;)
  {
    disparity.setTo(0);
    if(method==0)
    {
      cv::Ptr<cv::StereoBM> stereo= cv::StereoBM::create(/*ndisparities=*/n_disp, /*blockSize =*/w_size);
      cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
      cv::cvtColor(frame2, gray2, CV_BGR2GRAY);
      stereo->compute(gray1, gray2, disparity);
    }
    else if(method==1)
    {
      cv::Ptr<cv::StereoSGBM> stereo= cv::StereoSGBM::create(/*minDisparity=*/1, /*numDisparities=*/n_disp, /*blockSize =*/w_size,
                  // /*int P1=*/0, /*int P2=*/0, /*int disp12MaxDiff=*/0,
                  /*int P1=*/8*3*w_size*w_size, /*int P2=*/32*3*w_size*w_size, /*int disp12MaxDiff=*/0,
                  /*int preFilterCap=*/0, /*int uniquenessRatio=*/0,
                  /*int speckleWindowSize=*/0, /*int speckleRange=*/0
                  /*, mode*/);
      stereo->compute(frame1, frame2, disparity);
    }

    cv::normalize(disparity, disparity, 0, 255, CV_MINMAX, CV_8U);

    cv::imshow("camera1", frame1);
    cv::imshow("camera2", frame2);
    cv::imshow("disparity", disparity);
    cv::imshow("projected", ProjectDepth(disparity));
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
