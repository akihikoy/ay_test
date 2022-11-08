//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_features.cpp
    \brief   Test feature detection methods in OpenCV to realsense data.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.08, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_features.out ros_rs_features.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
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
  if(event == CV_EVENT_LBUTTONDOWN)
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

int max_count=1000;
bool using_feat_track_rgb=true;
bool using_corner_det_rgb=true;
bool using_feat_track_depth=true;
bool using_corner_det_depth=true;
int block_size_rgb=3;
double quality_level_rgb=0.01;
double min_dist_rgb=10;
int block_size_depth=3;
double quality_level_depth=0.01;
double min_dist_depth=10;
float dim_rgb=0.5;
float dim_points_rgb=1.0;
float dim_depth=0.5;
float dim_points_depth=1.0;

bool quit_at_cap_err(false);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  std::vector<cv::Point2f> points_rgb, points_depth;

  {
    cv::Mat gray;
    cv::cvtColor(frame_rgb, gray, CV_BGR2GRAY);
    if(using_feat_track_rgb)
      cv::goodFeaturesToTrack(gray, points_rgb, max_count, quality_level_rgb, min_dist_rgb, cv::Mat(), block_size_rgb, 0, 0.04);
    if(using_corner_det_rgb)
    {
      cv::Size subpix_win_size(10,10);
      cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
      cv::cornerSubPix(gray, points_rgb, subpix_win_size, cv::Size(-1,-1), termcrit);
    }
  }
  cv::Mat canvas_points_rgb(frame_rgb.size(),frame_rgb.type(),CV_RGB(0,0,0));
  for(std::vector<cv::Point2f>::const_iterator itr_pt(points_rgb.begin()),itr_pt_end(points_rgb.end());
      itr_pt!=itr_pt_end; ++itr_pt)
    cv::circle(canvas_points_rgb, *itr_pt, 2, CV_RGB(255,0,255));

  {
    cv::Mat gray;
    frame_depth.convertTo(gray, CV_32FC1);
    if(using_feat_track_depth)
      cv::goodFeaturesToTrack(gray, points_depth, max_count, quality_level_depth, min_dist_depth, cv::Mat(), block_size_depth, 0, 0.04);
    if(using_corner_det_depth)
    {
      cv::Size subpix_win_size(10,10);
      cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
      cv::cornerSubPix(gray, points_depth, subpix_win_size, cv::Size(-1,-1), termcrit);
    }
  }
  cv::Mat canvas_points_depth(frame_depth.size(),frame_rgb.type(),CV_RGB(0,0,0));
  for(std::vector<cv::Point2f>::const_iterator itr_pt(points_depth.begin()),itr_pt_end(points_depth.end());
      itr_pt!=itr_pt_end; ++itr_pt)
    cv::circle(canvas_points_rgb, *itr_pt, 2, CV_RGB(255,0,255));

  cv::Mat img_rgb_disp(frame_rgb*dim_rgb);
  img_rgb_disp+= dim_points_rgb*canvas_points_rgb;

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, CV_GRAY2BGR);
  img_depth_disp*= dim_depth;
  img_depth_disp+= dim_points_depth*canvas_points_depth;

  cv::imshow("depth", img_depth_disp);
  ProcMouseEvent("depth", frame_depth);

  cv::imshow("rgb", img_rgb_disp);
  ProcMouseEvent("rgb", frame_rgb);

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
  CreateTrackbar<int>("max_count", "depth", &max_count, 0, 10000, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("using_feat_track_depth", "depth", &using_feat_track_depth, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("using_corner_det_depth", "depth", &using_corner_det_depth, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("block_size_depth", "depth", &block_size_depth, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("quality_level_depth", "depth", &quality_level_depth, 0.0, 1.0, 0.001,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("min_dist_depth", "depth", &min_dist_depth, 0.0, 100.0, 0.1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("dim_depth", "depth", &dim_depth, 0.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("dim_points_depth", "depth", &dim_points_depth, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("rgb",1);
  setMouseCallback("rgb", onMouse, "rgb");
  CreateTrackbar<int>("max_count", "rgb", &max_count, 0, 10000, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("using_feat_track_rgb", "rgb", &using_feat_track_rgb, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("using_corner_det_rgb", "rgb", &using_corner_det_rgb, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("block_size_rgb", "rgb", &block_size_rgb, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("quality_level_rgb", "rgb", &quality_level_rgb, 0.0, 1.0, 0.001,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("min_dist_rgb", "rgb", &min_dist_rgb, 0.0, 100.0, 0.1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("dim_rgb", "rgb", &dim_rgb, 0.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("dim_points_rgb", "rgb", &dim_points_rgb, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
