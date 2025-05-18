//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_diff.cpp
    \brief   Compute a difference between two depth/rgb images at different times.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.11, 2023

$ g++ -O2 -g -W -Wall -o ros_rs_diff.out ros_rs_diff.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#define LIBRARY
#include "cv2-edge_cmp.cpp"
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"
//-------------------------------------------------------------------------------------------

bool mouse_event_detected(false);
int x_mouse(0), y_mouse(0);
std::string win_mouse("");

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
  if(event == cv::EVENT_LBUTTONDOWN)
  {
    mouse_event_detected= true;
    x_mouse= x; y_mouse= y;
  }
}


double depth_scale(0.3);
bool quit_at_cap_err(false);
cv::Mat FrameDepthRef, FrameRGBRef;

int threshold_rgb(50);
int threshold_hsv(50);
int n_erode(2), n_dilate(2);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  if(FrameDepthRef.empty() || mouse_event_detected)
    FrameDepthRef= frame_depth.clone();
  if(FrameRGBRef.empty() || mouse_event_detected)
    FrameRGBRef= frame_rgb.clone();
  mouse_event_detected= false;

  cv::Mat diff_depth, mask_depth;
  cv::absdiff(FrameDepthRef, frame_depth, diff_depth);
  mask_depth= diff_depth;

  cv::Mat diff_rgb, mask_rgb;
  cv::absdiff(FrameRGBRef, frame_rgb, diff_rgb);
  mask_rgb= cv::Mat::zeros(diff_rgb.rows, diff_rgb.cols, CV_8U);
  {
    float dist;
    for(int r(0),r_end(diff_rgb.rows); r<r_end; ++r)
      for(int c(0),c_end(diff_rgb.cols); c<c_end; ++c)
      {
        cv::Vec3b pt= diff_rgb.at<cv::Vec3b>(r,c);
        dist= std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
        // mask_rgb.at<unsigned char>(r,c)= std::min(255,int(dist));
        if(dist>threshold_rgb)  mask_rgb.at<unsigned char>(r,c)= 255;
      }
    if(n_erode>0)   cv::erode(mask_rgb,mask_rgb,cv::Mat(),cv::Point(-1,-1), n_erode);
    if(n_dilate>0)  cv::dilate(mask_rgb,mask_rgb,cv::Mat(),cv::Point(-1,-1), n_dilate);
  }

  cv::Mat frame_hsv, frame_ref_hsv, diff_hsv, mask_hsv;
  cv::cvtColor(frame_rgb, frame_hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(FrameRGBRef, frame_ref_hsv, cv::COLOR_BGR2HSV);
  cv::absdiff(frame_ref_hsv, frame_hsv, diff_hsv);
  mask_hsv= cv::Mat::zeros(diff_hsv.rows, diff_hsv.cols, CV_8U);
  {
    float dist;
    for(int r(0),r_end(diff_hsv.rows); r<r_end; ++r)
      for(int c(0),c_end(diff_hsv.cols); c<c_end; ++c)
      {
        cv::Vec3b pt= diff_hsv.at<cv::Vec3b>(r,c);
        dist= std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] +  pt[2]*pt[2]);
        // mask_hsv.at<unsigned char>(r,c)= std::min(255,int(dist));
        if(dist>threshold_hsv)  mask_hsv.at<unsigned char>(r,c)= 255;
      }
    if(n_erode>0)   cv::erode(mask_hsv,mask_hsv,cv::Mat(),cv::Point(-1,-1), n_erode);
    if(n_dilate>0)  cv::dilate(mask_hsv,mask_hsv,cv::Mat(),cv::Point(-1,-1), n_dilate);
  }

  cv::Mat frame_depth_disp(frame_depth*depth_scale);
  frame_depth_disp.convertTo(frame_depth_disp, CV_8U);
  cv::cvtColor(frame_depth_disp, frame_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("input_depth", frame_depth_disp);

  cv::Mat diff_depth_disp(diff_depth*depth_scale);
  diff_depth_disp.convertTo(diff_depth_disp, CV_8U);
  cv::cvtColor(diff_depth_disp, diff_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("diff_depth", diff_depth_disp);

  cv::Mat mask_depth_disp(diff_depth*depth_scale);
  mask_depth_disp.convertTo(mask_depth_disp, CV_8U);
  cv::cvtColor(mask_depth_disp, mask_depth_disp, cv::COLOR_GRAY2BGR);
  cv::imshow("mask_depth", mask_depth_disp);

  cv::imshow("input_rgb", frame_rgb);
  cv::imshow("diff_rgb", diff_rgb);
  cv::imshow("mask_rgb", mask_rgb);

  cv::imshow("diff_hsv", diff_hsv);
  cv::imshow("mask_hsv", mask_hsv);


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
  cv::setMouseCallback("input_depth", onMouse, NULL);
  cv::namedWindow("input_rgb",1);
  cv::setMouseCallback("input_rgb", onMouse, NULL);

  cv::namedWindow("diff_depth",1);
  cv::namedWindow("mask_depth",1);
  cv::namedWindow("diff_rgb",1);
  cv::namedWindow("mask_rgb",1);
  cv::namedWindow("diff_hsv",1);
  cv::namedWindow("mask_hsv",1);

  CreateTrackbar<int>("threshold_rgb", "mask_rgb", &threshold_rgb, 0, 255, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("threshold_hsv", "mask_hsv", &threshold_hsv, 0, 255, 1,  &TrackbarPrintOnTrack);

  CreateTrackbar<int>("n_erode", "mask_rgb", &n_erode, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_dilate", "mask_rgb", &n_dilate, 0, 50, 1,  &TrackbarPrintOnTrack);

  CreateTrackbar<int>("n_erode", "mask_hsv", &n_erode, 0, 50, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_dilate", "mask_hsv", &n_dilate, 0, 50, 1,  &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
