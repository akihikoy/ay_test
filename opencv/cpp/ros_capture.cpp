//-------------------------------------------------------------------------------------------
/*! \file    ros_capture.cpp
    \brief   Capture example from ROS image topics;
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021

$ g++ -O2 -g -W -Wall -o ros_capture.out ros_capture.cpp  -I../include -I/opt/ros/kinetic/include -pthread -llog4cxx -lpthread -L/opt/ros/kinetic/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -Wl,-rpath,/opt/ros/kinetic/lib

$ ./ros_capture.out /camera/color/image_raw
$ ./ros_capture.out /camera/aligned_depth_to_color/image_raw 16UC1

NOTE: If you experience segmentation fault, do this:
$ sudo apt-get -f install libcv-bridge0d
$ cd /opt/ros/kinetic/lib/
$ sudo mv libcv_bridge.so{,.trouble}
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libcv_bridge.so.0d libcv_bridge.so
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/bind.hpp>
//-------------------------------------------------------------------------------------------

typedef void(*TCVCallback)(const cv::Mat&);

void ImageCallback(const sensor_msgs::ImageConstPtr& msg, const std::string &encoding, TCVCallback callback, bool debug)
{
  if(debug)
  {
    std::cerr<<"msg->header: "<<msg->header<<std::endl;
    std::cerr<<"msg->height: "<<msg->height<<std::endl;
    std::cerr<<"msg->width: "<<msg->width<<std::endl;
    std::cerr<<"msg->encoding: "<<msg->encoding<<std::endl;
    std::cerr<<"msg->is_bigendian: "<<int(msg->is_bigendian)<<std::endl;
    std::cerr<<"msg->step: "<<msg->step<<std::endl;
    std::cerr<<"msg->data.size(): "<<msg->data.size()<<std::endl;
    std::cerr<<"msg->data[0,1,2]: "<<int(msg->data[0])<<" "<<int(msg->data[1])<<" "<<int(msg->data[2])<<std::endl;
    std::cerr<<"msg->data[size()-1]: "<<int(msg->data[msg->data.size()-1])<<std::endl;
    std::cerr<<"msg->data[size()-2]: "<<int(msg->data[msg->data.size()-2])<<std::endl;
    std::cerr<<"msg->data[size()-3]: "<<int(msg->data[msg->data.size()-3])<<std::endl;
  }
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, encoding);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  if(debug) std::cerr<<"cv_ptr: "<<cv_ptr<<std::endl;
  // std::cerr<<"cv_ptr->image: "<<cv_ptr->image<<std::endl;
  cv::Mat frame= cv_ptr->image;
  callback(frame);
}
//-------------------------------------------------------------------------------------------

void FinishLoop()
{
  ros::shutdown();
}
//-------------------------------------------------------------------------------------------

void StartLoop(int argc, char**argv, const std::string &img_topic, const std::string &encoding, TCVCallback callback, const std::string &node_name="img_node")
{
  ros::init(argc, argv, node_name);
  ros::NodeHandle node("~");

  ros::Subscriber sub_img= node.subscribe<sensor_msgs::Image>(img_topic, 1, boost::bind(&ImageCallback,_1,encoding,callback,false));

  ros::spin();
}
//-------------------------------------------------------------------------------------------


#ifndef LIBRARY
void CVCallback(const cv::Mat &frame)
{
  cv::imshow("camera", frame);
  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/color/image_raw"), encoding(sensor_msgs::image_encodings::BGR8);
  if(argc>1)  img_topic= argv[1];
  if(argc>2)  encoding= argv[2];
  cv::namedWindow("camera",1);
  StartLoop(argc, argv, img_topic, encoding, CVCallback);
  return 0;
}
//-------------------------------------------------------------------------------------------
#endif//LIBRARY
