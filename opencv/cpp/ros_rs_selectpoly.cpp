//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_selectpoly.cpp
    \brief   Base of test for depth/rgb/normal/point images with a polygon selection.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.22, 2023

$ g++ -O2 -g -W -Wall -o ros_rs_selectpoly.out ros_rs_selectpoly.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
#define LIBRARY
#include "ros_proj_mat.cpp"
#include "ros_rs_normal.cpp"
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"
//-------------------------------------------------------------------------------------------

std::vector<std::vector<cv::Point> >  polygon(1);

bool mouse_event_detected(false);
int x_mouse(0), y_mouse(0), mouse_event(0), mouse_flags(0);

void OnMouse(int event, int x, int y, int flags, void *)
{
  if(event==cv::EVENT_LBUTTONDOWN || event==cv::EVENT_LBUTTONUP
    || event==cv::EVENT_RBUTTONDOWN || event==cv::EVENT_RBUTTONUP)
  {
    mouse_event_detected= true;
    x_mouse= x; y_mouse= y;
    mouse_event= event;
    mouse_flags= flags;
  }
}
//-------------------------------------------------------------------------------------------

void ProcMouseEvent(
    const cv::Mat &frame_rgb, const cv::Mat &frame_depth,
    const cv::Mat &normal_img, const cv::Mat &cloud_img)
{
  if(!mouse_event_detected)  return;

  if(mouse_event==cv::EVENT_LBUTTONDOWN/* && mouse_flags==0*/)
  {
    std::cout<<"Clicked:["<<x_mouse<<","<<y_mouse<<"]:"
        <<" RGB:"<<GetPixelVal(frame_rgb,x_mouse,y_mouse)
        <<" Depth:"<<GetPixelVal(frame_depth,x_mouse,y_mouse)
        <<" Normal:"<<GetPixelVal(normal_img,x_mouse,y_mouse)
        <<" Pt3d:"<<GetPixelVal(cloud_img,x_mouse,y_mouse)
        <<std::endl;
  }
  if(mouse_event==cv::EVENT_LBUTTONDOWN && mouse_flags&cv::EVENT_FLAG_SHIFTKEY)
  {
    polygon[0].push_back(cv::Point(x_mouse,y_mouse));
  }
  if(mouse_event==cv::EVENT_RBUTTONDOWN && mouse_flags&cv::EVENT_FLAG_SHIFTKEY)
  {
    std::cout<<"Flushing polygon:";
    for(size_t i(0); i<polygon[0].size(); ++i)
      std::cout<<" "<<polygon[0][i];
    std::cout<<std::endl;
    polygon[0].clear();
  }
  mouse_event_detected= false;
}
//-------------------------------------------------------------------------------------------

double depth_scale(0.3);
bool quit_at_cap_err(false);
cv::Mat proj_mat;

int wsize(5);
int cd2ntype(cd2ntSimple);
float resize_ratio(0.5);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  cv::Mat normal_img, cloud_img;
  DepthImgToNormalImg(
    frame_depth, proj_mat,
    normal_img, cloud_img, wsize, resize_ratio, /*type=*/TCD2NType(cd2ntype));  // cd2ntSimple,cd2ntRobust

  if(mouse_event_detected)
  {
    ProcMouseEvent(frame_rgb, frame_depth, normal_img, cloud_img);
// process something...
  }

  cv::polylines(frame_rgb, polygon, /*isClosed=*/true, CV_RGB(255,0,255), 2);
  cv::polylines(frame_depth, polygon, /*isClosed=*/true, 800, 2);
  cv::polylines(normal_img, polygon, /*isClosed=*/true, cv::Vec3f(1,0,1), 2);
  cv::polylines(cloud_img, polygon, /*isClosed=*/true, cv::Vec3f(1,0,1), 2);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, CV_GRAY2BGR);
  cv::imshow("input_depth", img_depth_disp);

  cv::imshow("input_rgb", frame_rgb);

  cv::imshow("normal", normal_img);
  cv::imshow("points3d", cloud_img);

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

  // cv::Mat proj_mat;
  std::string img_info_topic(img1_topic), frame_id, ltopic("image_raw");
  img_info_topic.replace(img_info_topic.find(ltopic),ltopic.length(),"camera_info");
  GetCameraProjectionMatrix(img_info_topic, frame_id, proj_mat);

  cv::namedWindow("input_depth",1);
  cv::setMouseCallback("input_depth", OnMouse);
  cv::namedWindow("input_rgb",1);
  cv::setMouseCallback("input_rgb", OnMouse);
  cv::namedWindow("normal",1);
  cv::setMouseCallback("normal", OnMouse);
  cv::namedWindow("points3d",1);
  cv::setMouseCallback("points3d", OnMouse);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
