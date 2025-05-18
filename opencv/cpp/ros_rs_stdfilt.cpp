//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_stdfilt.cpp
    \brief   Test of stdfilt.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.08, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_stdfilt.out ros_rs_stdfilt.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4

ref. https://stackoverflow.com/questions/7331105/stdfilt-in-opencv
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

void StdFilt(const cv::Mat img, cv::Mat &filtered, int kernel_size=3)
{
  cv::Mat h=cv::Mat::ones(kernel_size,kernel_size,CV_32FC1);
  float n=cv::sum(h)[0];
  float n1=n-1;
  std::cerr<<"debug:sum(img)="<<cv::sum(img)[0];
  cv::Mat img_sq=img.mul(img);
  std::cerr<<", sum(img_sq)="<<cv::sum(img_sq)[0];
  cv::Mat c1,c2;
  cv::filter2D(img_sq, c1, /*ddepth=*/-1, /*kernel=*/h/n1, /*anchor=*/cv::Point(-1,-1), /*delta=*/0.0, /*borderType=*/cv::BORDER_REFLECT);
  cv::filter2D(img, c2, /*ddepth=*/-1, /*kernel=*/h, /*anchor=*/cv::Point(-1,-1), /*delta=*/0.0, /*borderType=*/cv::BORDER_REFLECT);
  std::cerr<<", sum(c1)="<<cv::sum(c1)[0];
  std::cerr<<", sum(c2)="<<cv::sum(c2)[0];
  cv::Mat c2_sq=c2.mul(c2) / (n*n1);
  std::cerr<<", sum(c2_sq)="<<cv::sum(c2_sq)[0];
  cv::Mat d=c1-c2_sq;
  std::cerr<<", sum(c1-c2_sq)="<<cv::sum(d)[0]<<std::endl;
  cv::max(d,0,d);
  std::cerr<<", sum(d)="<<cv::sum(d)[0]<<std::endl;
  cv::sqrt(d, filtered);
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
#define LIBRARY
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"
//-------------------------------------------------------------------------------------------

bool mouse_event_detected(false);
int x_mouse(0), y_mouse(0);
std::string win_mouse("");
void setMouseCallback(const std::string &winname, cv::MouseCallback onMouse, const char *userdata)
{
  cv::setMouseCallback(winname, onMouse, const_cast<char*>(userdata));
}
static void onMouse(int event, int x, int y, int /*flags*/, void* param)
{
  if(event == cv::EVENT_LBUTTONDOWN)
  {
    mouse_event_detected= true;
    x_mouse= x; y_mouse= y;
    win_mouse= std::string(reinterpret_cast<const char*>(param));
  }
}
void ProcMouseEvent(const std::string &win, const cv::Mat &m)
{
  if(mouse_event_detected && win_mouse==win)
  {
    std::cout<<win<<": clicked: ("<<x_mouse<<","<<y_mouse<<"): value= "<<GetPixelVal(m,x_mouse,y_mouse)<<std::endl;
    mouse_event_detected= false;
  }
}
//-------------------------------------------------------------------------------------------

double depth_scale(0.3);

int kernel_size_rgb=3;
int kernel_size_depth=3;
float scale_rgb=1.0;
float scale_depth=1.0;

bool quit_at_cap_err(false);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  cv::Mat stdfilt_rgb;
  {
    cv::Mat gray, grayn;
    cv::cvtColor(frame_rgb, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(grayn, CV_32FC1);
    grayn/=255.0;
    StdFilt(grayn, stdfilt_rgb, kernel_size_rgb);
  }
  cv::Mat stdfilt_rgb_disp(scale_rgb*stdfilt_rgb*255.0);
  stdfilt_rgb_disp.convertTo(stdfilt_rgb_disp, CV_8U);
  cv::cvtColor(stdfilt_rgb_disp, stdfilt_rgb_disp, cv::COLOR_GRAY2BGR);

  cv::Mat stdfilt_depth;
  {
    cv::Mat grayn;
    frame_depth.convertTo(grayn, CV_32FC1);
    grayn/=750.0;
    StdFilt(grayn, stdfilt_depth, kernel_size_depth);
  }
  cv::Mat stdfilt_depth_disp(scale_depth*stdfilt_depth*255.0);
  stdfilt_depth_disp.convertTo(stdfilt_depth_disp, CV_8U);
  cv::cvtColor(stdfilt_depth_disp, stdfilt_depth_disp, cv::COLOR_GRAY2BGR);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("depth", img_depth_disp);
  ProcMouseEvent("depth", frame_depth);

  cv::imshow("rgb", frame_rgb);
  ProcMouseEvent("rgb", frame_rgb);

  cv::imshow("stdfilt_depth", /*dim_depth*frame_depth+*/stdfilt_depth_disp);
  ProcMouseEvent("stdfilt_depth", stdfilt_depth);

  cv::imshow("stdfilt_rgb", /*dim_rgb*frame_rgb+*/stdfilt_rgb_disp);
  ProcMouseEvent("stdfilt_rgb", stdfilt_rgb);

  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img1_topic("/camera/aligned_depth_to_color/image_raw");
  std::string img2_topic("/camera/color/image_raw");
  if(argc>1)  img1_topic= argv[1];
  if(argc>2)  img2_topic= argv[2];
  ros::init(argc, argv, "ros_rs_edge_cmp");
  ros::NodeHandle node("~");
  std::string encoding1= GetImageEncoding(img1_topic, node, /*convert_cv=*/true);
  std::string encoding2= GetImageEncoding(img2_topic, node, /*convert_cv=*/true);
  if(encoding1!="16UC1")
  {
    std::cerr<<"WARNING: We assume img1 as a depth image topic, but is "<<encoding1<<std::endl;
  }

  cv::namedWindow("depth",1);
  setMouseCallback("depth", onMouse, "depth");

  cv::namedWindow("stdfilt_depth",1);
  setMouseCallback("stdfilt_depth", onMouse, "stdfilt_depth");
  CreateTrackbar<int>("kernel_size_depth", "stdfilt_depth", &kernel_size_depth, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("scale_depth", "stdfilt_depth", &scale_depth, 0.0f, 100.0f, 0.1f,  &TrackbarPrintOnTrack);

  cv::namedWindow("rgb",1);
  setMouseCallback("rgb", onMouse, "rgb");

  cv::namedWindow("stdfilt_rgb",1);
  setMouseCallback("stdfilt_rgb", onMouse, "stdfilt_rgb");
  CreateTrackbar<int>("kernel_size_rgb", "stdfilt_rgb", &kernel_size_rgb, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("scale_rgb", "stdfilt_rgb", &scale_rgb, 0.0f, 100.0f, 0.1f,  &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
