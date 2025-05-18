//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_planecol.cpp
    \brief   Extract RGB image according to the corresponding depth.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.09, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_planecol.out ros_rs_planecol.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------

// Extract the input image according to the values of corresponding depth pixels.
void ExtractImgByDepthRange(const cv::Mat &img_in, const cv::Mat &img_depth,cv::Mat &img_out, int min, int max, const cv::Scalar &fill=cv::Scalar(255,255,255))
{
  img_out.create(img_in.size(), img_in.type());
  img_out.setTo(fill);
  cv::Mat mask_depth;
  cv::inRange(img_depth, min, max, mask_depth);
  img_in.copyTo(img_out, mask_depth);
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
#ifndef LIBRARY
#include <unistd.h>
#include "cap_open.h"
#define LIBRARY
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"

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

int depth_min=1;
int depth_max=709;

bool quit_at_cap_err(false);


void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  cv::Mat modified_rgb;
  ExtractImgByDepthRange(frame_rgb, frame_depth, modified_rgb, depth_min, depth_max);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("input_depth", img_depth_disp);
  ProcMouseEvent("input_depth", frame_depth);

  cv::imshow("input_rgb", frame_rgb);
  ProcMouseEvent("input_rgb", frame_rgb);

  cv::imshow("modified_rgb", modified_rgb);
  ProcMouseEvent("modified_rgb", modified_rgb);


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

  cv::namedWindow("modified_rgb",1);
  setMouseCallback("modified_rgb", onMouse, "modified_rgb");
  CreateTrackbar<int>("depth_min", "modified_rgb", &depth_min, 0, 2000, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("depth_max", "modified_rgb", &depth_max, 0, 2000, 1,  &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
#endif  // LIBRARY
//-------------------------------------------------------------------------------------------
