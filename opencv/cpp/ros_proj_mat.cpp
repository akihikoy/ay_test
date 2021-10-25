//-------------------------------------------------------------------------------------------
/*! \file    ros_proj_mat.cpp
    \brief   Get projection matrix of a camera from ROS topics;
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.14, 2021

g++ -O2 -g -W -Wall -o ros_proj_mat.out ros_proj_mat.cpp  -I../include -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
//-------------------------------------------------------------------------------------------

// Get camera projection matrix from ros topic.
void GetCameraProjectionMatrix(const std::string &cam_info_topic, std::string &frame_id, cv::Mat &proj_mat)
{
  ros::NodeHandle node("~");
  boost::shared_ptr<sensor_msgs::CameraInfo const> ptr_cam_info;
  sensor_msgs::CameraInfo cam_info;
  ptr_cam_info= ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic, node);
  if(ptr_cam_info==NULL)
  {
    std::cerr<<"Failed to get camera info from the topic: "<<cam_info_topic<<std::endl;
    return;
  }
  cam_info= *ptr_cam_info;

  // std::cerr<<"cam_info: "<<cam_info<<std::endl;
  frame_id= cam_info.header.frame_id;
  // cv::Mat proj_mat(3,4, CV_64F, cam_info.P);
  proj_mat.create(3,4, CV_64F);
  for(int r(0),i(0);r<3;++r)
    for(int c(0);c<4;++c,++i)
      proj_mat.at<double>(r,c)= cam_info.P[i];
}
//-------------------------------------------------------------------------------------------


#ifndef LIBRARY
int main(int argc, char**argv)
{
  std::string cam_info_topic("/camera/aligned_depth_to_color/camera_info");
  if(argc>1)  cam_info_topic= argv[1];

  std::string node_name="test_node";

  ros::init(argc, argv, node_name);
  ros::NodeHandle node("~");


  std::string frame_id;
  cv::Mat proj_mat;
  GetCameraProjectionMatrix(cam_info_topic, frame_id, proj_mat);
  std::cerr<<"frame_id: "<<frame_id<<std::endl;
  std::cerr<<"proj_mat: "<<proj_mat<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
#endif//LIBRARY
