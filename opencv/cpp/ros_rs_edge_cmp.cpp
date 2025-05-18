//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_edge_cmp.cpp
    \brief   Comparison of edge detection methods for RGB or depth from a RealSense camera.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.21, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_edge_cmp.out ros_rs_edge_cmp.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
#define LIBRARY
#include "cv2-edge_cmp.cpp"
#include "ros_capture.cpp"
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

bool is_depth(false);
double depth_scale(0.3);

double canny_threshold1=100.0;
double canny_threshold2=200.0;
int    canny_ksize=5;
int    canny_blur_size=3;
double canny_blur_std=1.5;

int    laplacian_ksize=5;
double laplacian_scale=2.0;
double laplacian_delta=0.0;
int    laplacian_blur_size=1;
double laplacian_blur_std=1.5;

int    sobel_ksize=3;
double sobel_scale=6.0;
double sobel_delta=0.0;
int    sobel_blur_size=1;
double sobel_blur_std=1.5;

bool quit_at_cap_err(false), disp_sum(false);
double sumscale(1.0e-6);

void CVCallback(const cv::Mat &frame)
{
  if(frame.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }
  cv::Mat canny, laplacian, sobel;

  canny= GetCanny(frame,
    canny_threshold1, canny_threshold2, canny_ksize,
    canny_blur_size, canny_blur_std,
    is_depth);
  laplacian= GetLaplacian(frame,
    laplacian_ksize, laplacian_scale, laplacian_delta,
    laplacian_blur_size, laplacian_blur_std,
    is_depth);
  sobel= GetSobel(frame,
    sobel_ksize, sobel_scale, sobel_delta,
    sobel_blur_size, sobel_blur_std,
    is_depth);

  if(!is_depth)
    cv::imshow("input", frame);
  else
  {
    cv::Mat img_disp(frame*depth_scale);
    img_disp.convertTo(img_disp, CV_8U);
    cv::cvtColor(img_disp, img_disp, cv::COLOR_GRAY2BGR);
    cv::imshow("input", img_disp);
  }
  cv::imshow("canny", canny);
  cv::imshow("laplacian", laplacian);
  cv::imshow("sobel", sobel);

  if(disp_sum)
    std::cout<<"canny, laplacian, sobel: "<<cv::sum(canny)*sumscale<<", "<<cv::sum(laplacian)*sumscale<<", "<<cv::sum(sobel)*sumscale<<std::endl;

  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
  if(c==' ')  disp_sum=!disp_sum;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/color/image_raw"), encoding;
  if(argc>1)  img_topic= argv[1];
  ros::init(argc, argv, "ros_rs_edge_cmp");
  ros::NodeHandle node("~");
  encoding= GetImageEncoding(img_topic, node, /*convert_cv=*/true);
  if(encoding=="16UC1")  is_depth= true;

  cv::namedWindow("input",1);

  cv::namedWindow("canny",1);
  CreateTrackbar<double>("threshold1", "canny", &canny_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny", &canny_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("ksize",      "canny", &canny_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size",  "canny", &canny_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std",   "canny", &canny_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("laplacian",1);
  CreateTrackbar<int>   ("ksize",    "laplacian", &laplacian_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "laplacian", &laplacian_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "laplacian", &laplacian_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","laplacian", &laplacian_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian", &laplacian_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("sobel",1);
  CreateTrackbar<int>   ("ksize",    "sobel", &sobel_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "sobel", &sobel_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "sobel", &sobel_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","sobel", &sobel_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel", &sobel_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img_topic, encoding, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
