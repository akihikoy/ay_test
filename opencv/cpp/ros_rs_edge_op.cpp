//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_edge_op.cpp
    \brief   Comparison of edge detection methods for RGB or depth from a RealSense camera.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.24, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_edge_op.out ros_rs_edge_op.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4
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


double depth_scale(0.3);

int depth_min=1;
int depth_max=707;

double canny_depth_threshold1=100.0;
double canny_depth_threshold2=200.0;
int    canny_depth_ksize=5;
int    canny_depth_blur_size=3;
double canny_depth_blur_std=1.5;

double canny_rgb_threshold1=100.0;
double canny_rgb_threshold2=200.0;
int    canny_rgb_ksize=5;
int    canny_rgb_blur_size=3;
double canny_rgb_blur_std=1.5;

int    laplacian_depth_ksize=5;
double laplacian_depth_scale=2.0;
double laplacian_depth_delta=0.0;
int    laplacian_depth_blur_size=1;
double laplacian_depth_blur_std=1.5;

int    laplacian_rgb_ksize=5;
double laplacian_rgb_scale=2.0;
double laplacian_rgb_delta=0.0;
int    laplacian_rgb_blur_size=1;
double laplacian_rgb_blur_std=1.5;

int    sobel_depth_ksize=3;
double sobel_depth_scale=6.0;
double sobel_depth_delta=0.0;
int    sobel_depth_blur_size=1;
double sobel_depth_blur_std=1.5;

int    sobel_rgb_ksize=3;
double sobel_rgb_scale=6.0;
double sobel_rgb_delta=0.0;
int    sobel_rgb_blur_size=1;
double sobel_rgb_blur_std=1.5;

int edge_kind_depth(2);  //0:canny,1:laplacian,2:sobel
int edge_threshold_depth(100);

int edge_kind_rgb(2);  //0:canny,1:laplacian,2:sobel
int edge_threshold_rgb(100);

bool quit_at_cap_err(false);
double sumscale(1.0e-6);

bool subtract_edge_depth(false);
bool subtract_edge_rgb(false);
bool subtract_edge_and(false);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  cv::Mat depth_range;
  cv::inRange(frame_depth, depth_min, depth_max, depth_range);

  cv::Mat canny_depth, laplacian_depth, sobel_depth;
  canny_depth= GetCanny(frame_depth,
    canny_depth_threshold1, canny_depth_threshold2, canny_depth_ksize,
    canny_depth_blur_size,  canny_depth_blur_std,
    /*is_depth=*/true);
  laplacian_depth= GetLaplacian(frame_depth,
    laplacian_depth_ksize,     laplacian_depth_scale, laplacian_depth_delta,
    laplacian_depth_blur_size, laplacian_depth_blur_std,
    /*is_depth=*/true);
  sobel_depth= GetSobel(frame_depth,
    sobel_depth_ksize,     sobel_depth_scale, sobel_depth_delta,
    sobel_depth_blur_size, sobel_depth_blur_std,
    /*is_depth=*/true);

  cv::Mat canny_rgb, laplacian_rgb, sobel_rgb;
  canny_rgb= GetCanny(frame_rgb,
    canny_rgb_threshold1, canny_rgb_threshold2, canny_rgb_ksize,
    canny_rgb_blur_size,  canny_rgb_blur_std);
  laplacian_rgb= GetLaplacian(frame_rgb,
    laplacian_rgb_ksize,     laplacian_rgb_scale, laplacian_rgb_delta,
    laplacian_rgb_blur_size, laplacian_rgb_blur_std);
  sobel_rgb= GetSobel(frame_rgb,
    sobel_rgb_ksize,     sobel_rgb_scale, sobel_rgb_delta,
    sobel_rgb_blur_size, sobel_rgb_blur_std);

  cv::Mat edge_mask_depth;
  if     (edge_kind_depth==0)
    cv::threshold(canny_depth,edge_mask_depth,edge_threshold_depth,255,cv::THRESH_BINARY);
  else if(edge_kind_depth==1)
    cv::threshold(laplacian_depth,edge_mask_depth,edge_threshold_depth,255,cv::THRESH_BINARY);
  else if(edge_kind_depth==2)
  {
    cv::Mat gray;
    cv::cvtColor(sobel_depth, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray,edge_mask_depth,edge_threshold_depth,255,cv::THRESH_BINARY);
  }

  cv::Mat edge_mask_rgb;
  if     (edge_kind_rgb==0)
    cv::threshold(canny_rgb,edge_mask_rgb,edge_threshold_rgb,255,cv::THRESH_BINARY);
  else if(edge_kind_rgb==1)
    cv::threshold(laplacian_rgb,edge_mask_rgb,edge_threshold_rgb,255,cv::THRESH_BINARY);
  else if(edge_kind_rgb==2)
  {
    cv::Mat gray;
    cv::cvtColor(sobel_rgb, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray,edge_mask_rgb,edge_threshold_rgb,255,cv::THRESH_BINARY);
  }

  cv::Mat edge_mask_and;
  cv::bitwise_and(edge_mask_depth,edge_mask_rgb, edge_mask_and);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("input_depth", img_depth_disp);
  ProcMouseEvent("input_depth", frame_depth);

  cv::imshow("input_rgb", frame_rgb);
  ProcMouseEvent("input_rgb", frame_rgb);

  cv::imshow("depth_range", depth_range);
  ProcMouseEvent("depth_range", depth_range);

  cv::imshow("canny_depth", canny_depth);
  cv::imshow("laplacian_depth", laplacian_depth);
  cv::imshow("sobel_depth", sobel_depth);
  ProcMouseEvent("canny_depth", canny_depth);
  ProcMouseEvent("laplacian_depth", laplacian_depth);
  ProcMouseEvent("sobel_depth", sobel_depth);

  cv::imshow("canny_rgb", canny_rgb);
  cv::imshow("laplacian_rgb", laplacian_rgb);
  cv::imshow("sobel_rgb", sobel_rgb);
  ProcMouseEvent("canny_rgb", canny_rgb);
  ProcMouseEvent("laplacian_rgb", laplacian_rgb);
  ProcMouseEvent("sobel_rgb", sobel_rgb);

  cv::imshow("edge_mask_depth", edge_mask_depth);
  cv::imshow("edge_mask_rgb", edge_mask_rgb);
  ProcMouseEvent("edge_mask_depth", edge_mask_depth);
  ProcMouseEvent("edge_mask_rgb", edge_mask_rgb);

  cv::imshow("edge_mask_and", edge_mask_and);
  ProcMouseEvent("edge_mask_and", edge_mask_and);

  cv::Mat edge_op;
  depth_range.copyTo(edge_op);
  if(subtract_edge_depth)  edge_op-= edge_mask_depth;
  if(subtract_edge_rgb)  edge_op-= edge_mask_rgb;
  if(subtract_edge_and)  edge_op-= edge_mask_and;
  cv::imshow("edge_op", edge_op);
  ProcMouseEvent("edge_op", edge_op);

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

  cv::namedWindow("input_depth",1);
  setMouseCallback("input_depth", onMouse, "input_depth");
  cv::namedWindow("input_rgb",1);
  setMouseCallback("input_rgb", onMouse, "input_rgb");

  cv::namedWindow("depth_range",1);
  setMouseCallback("depth_range", onMouse, "depth_range");
  CreateTrackbar<int>   ("depth_min", "depth_range", &depth_min, 0, 10000, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("depth_max", "depth_range", &depth_max, 0, 10000, 1,  &TrackbarPrintOnTrack);

  cv::namedWindow("canny_depth",1);
  setMouseCallback("canny_depth", onMouse, "canny_depth");
  CreateTrackbar<double>("threshold1", "canny_depth", &canny_depth_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny_depth", &canny_depth_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("ksize",      "canny_depth", &canny_depth_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size",  "canny_depth", &canny_depth_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std",   "canny_depth", &canny_depth_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("canny_rgb",1);
  setMouseCallback("canny_rgb", onMouse, "canny_rgb");
  CreateTrackbar<double>("threshold1", "canny_rgb", &canny_rgb_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny_rgb", &canny_rgb_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("ksize",      "canny_rgb", &canny_rgb_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size",  "canny_rgb", &canny_rgb_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std",   "canny_rgb", &canny_rgb_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("laplacian_depth",1);
  setMouseCallback("laplacian_depth", onMouse, "laplacian_depth");
  CreateTrackbar<int>   ("ksize",    "laplacian_depth", &laplacian_depth_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "laplacian_depth", &laplacian_depth_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "laplacian_depth", &laplacian_depth_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","laplacian_depth", &laplacian_depth_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian_depth", &laplacian_depth_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("laplacian_rgb",1);
  setMouseCallback("laplacian_rgb", onMouse, "laplacian_rgb");
  CreateTrackbar<int>   ("ksize",    "laplacian_rgb", &laplacian_rgb_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "laplacian_rgb", &laplacian_rgb_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "laplacian_rgb", &laplacian_rgb_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","laplacian_rgb", &laplacian_rgb_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian_rgb", &laplacian_rgb_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("sobel_depth",1);
  setMouseCallback("sobel_depth", onMouse, "sobel_depth");
  CreateTrackbar<int>   ("ksize",    "sobel_depth", &sobel_depth_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "sobel_depth", &sobel_depth_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "sobel_depth", &sobel_depth_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","sobel_depth", &sobel_depth_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel_depth", &sobel_depth_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("sobel_rgb",1);
  setMouseCallback("sobel_rgb", onMouse, "sobel_rgb");
  CreateTrackbar<int>   ("ksize",    "sobel_rgb", &sobel_rgb_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "sobel_rgb", &sobel_rgb_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "sobel_rgb", &sobel_rgb_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","sobel_rgb", &sobel_rgb_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel_rgb", &sobel_rgb_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("edge_mask_depth",1);
  setMouseCallback("edge_mask_depth", onMouse, "edge_mask_depth");
  CreateTrackbar<int>   ("edge_kind",      "edge_mask_depth", &edge_kind_depth, 0, 2, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("edge_threshold", "edge_mask_depth", &edge_threshold_depth, 0, 255, 1,  &TrackbarPrintOnTrack);

  cv::namedWindow("edge_mask_rgb",1);
  setMouseCallback("edge_mask_rgb", onMouse, "edge_mask_rgb");
  CreateTrackbar<int>   ("edge_kind",      "edge_mask_rgb", &edge_kind_rgb, 0, 2, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("edge_threshold", "edge_mask_rgb", &edge_threshold_rgb, 0, 255, 1,  &TrackbarPrintOnTrack);

  cv::namedWindow("edge_mask_and",1);
  setMouseCallback("edge_mask_and", onMouse, "edge_mask_and");

  cv::namedWindow("edge_op",1);
  setMouseCallback("edge_op", onMouse, "edge_op");
  CreateTrackbar<bool>  ("subtract_edge_depth", "edge_op", &subtract_edge_depth, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>  ("subtract_edge_rgb", "edge_op", &subtract_edge_rgb, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>  ("subtract_edge_and", "edge_op", &subtract_edge_and, &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
