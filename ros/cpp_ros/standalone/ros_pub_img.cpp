//-------------------------------------------------------------------------------------------
/*! \file    ros_pub_img.cpp
    \brief   Capture images from a camera and publish them as ROS image topics.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.27, 2021

ref. For an advanced version, see:
ay_vision/ay_vision/src_ros/cv_usb_node.cpp

$ g++ -O2 -g -W -Wall -o ros_pub_img.out ros_pub_img.cpp  -I/opt/ros/kinetic/include -pthread -llog4cxx -lpthread -L/opt/ros/kinetic/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lboost_system -limage_transport -lcamera_info_manager -Wl,-rpath,/opt/ros/kinetic/lib

$ ./ros_pub_img.out
$ ./ros_capture.out
*/
//-------------------------------------------------------------------------------------------
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <camera_info_manager/camera_info_manager.h>
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  ros::init(argc, argv, "ros_pub_img");
  ros::NodeHandle node("~");

  image_transport::ImageTransport imgtr(node);
  image_transport::Publisher pub= imgtr.advertise("/camera/color/image_raw", 1);
  camera_info_manager::CameraInfoManager info_manager(ros::NodeHandle("~/camera"), "camera", /*camera_info_url=*/"");

  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;ros::ok();)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    pub.publish( cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg() );

    cv::imshow("camera", frame);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
