//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_cache.cpp
    \brief   Test of message_filters::Cache
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.02, 2025

$ g++ -O2 -g -W -Wall -o ros_rs_cache.out ros_rs_cache.cpp  -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lmessage_filters -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <boost/bind.hpp>
//-------------------------------------------------------------------------------------------

typedef void(*TCVCallback)(const cv::Mat&, const std_msgs::Header&, const cv::Mat&, const std_msgs::Header&);

// Get the image encoding of a ROS image topic.
// If convert_cv is true, the encoding is converted for OpenCV image conversion.
std::string GetImageEncoding(const std::string &img_topic, ros::NodeHandle &node, bool convert_cv=false, const double &time_out=5.0)
{
  sensor_msgs::ImageConstPtr ptr_img_header;
  ptr_img_header= ros::topic::waitForMessage<sensor_msgs::Image>(img_topic, node, ros::Duration(time_out));
  if(ptr_img_header==NULL)
  {
    std::cerr<<"Failed to receive the image topic: "<<img_topic<<std::endl;
    return "";
  }
  const std::string &encoding(ptr_img_header->encoding);
  if(!convert_cv)  return encoding;
  if(encoding=="rgb8")  return "bgr8";
  if(encoding=="RGB8")  return "BGR8";
  // TODO: Add more conversion if necessary.
  return encoding;
}
//-------------------------------------------------------------------------------------------

struct TConvImgMsgRes
{
  std_msgs::Header Header;
  cv::Mat Frame;
};

TConvImgMsgRes ConvImgMsg(const sensor_msgs::ImageConstPtr& msg, const std::string &encoding)
{
  TConvImgMsgRes res;
  res.Header= msg->header;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, encoding);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return res;
  }
  res.Frame= cv_ptr->image;
  return res;
}
//-------------------------------------------------------------------------------------------

// Pick the cached image closest to t without allocating vectors.
// Returns nullptr if no element lies within [t - slop, t + slop].
sensor_msgs::ImageConstPtr FindClosestCache(message_filters::Cache<sensor_msgs::Image>& cache, const ros::Time& t)
{
  auto before= cache.getElemBeforeTime(t);  // newest before t
  auto after = cache.getElemAfterTime(t);   // oldest after t

  if(!before && !after) return sensor_msgs::ImageConstPtr();
  if(before && !after)  return before;
  if(after && !before)  return after;
  if(t-before->header.stamp <= after->header.stamp-t)  return before;
  return after;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img1_topic("/camera/aligned_depth_to_color/image_raw");
  std::string img2_topic("/camera/color/image_raw");
  std::string encoding1(""/*sensor_msgs::image_encodings::TYPE_16UC1*/);
  std::string encoding2(""/*sensor_msgs::image_encodings::BGR8*/);
  std::string node_name= "test_node";
  if(argc>1)  img1_topic= argv[1];
  if(argc>2)  img2_topic= argv[2];
  if(argc>3)  encoding1= argv[3];
  if(argc>4)  encoding2= argv[4];
  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);

  if(node_name!="")  ros::init(argc, argv, node_name);
  ros::NodeHandle node("~");
  if(encoding1=="")  encoding1= GetImageEncoding(img1_topic, node, /*convert_cv=*/true);
  if(encoding2=="")  encoding2= GetImageEncoding(img2_topic, node, /*convert_cv=*/true);

  message_filters::Subscriber<sensor_msgs::Image> sub1, sub2;
  message_filters::Cache<sensor_msgs::Image> cache1, cache2;

  sub1.subscribe(node, img1_topic, 10);
  sub2.subscribe(node, img2_topic, 10);

  cache1.connectInput(sub1);
  cache1.setCacheSize(60);

  cache2.connectInput(sub2);
  cache2.setCacheSize(60);

  while(ros::ok())
  {
    ros::spinOnce();
    auto msg1= FindClosestCache(cache1, ros::Time::now() - ros::Duration(0.0));
    auto msg2= FindClosestCache(cache2, ros::Time::now() - ros::Duration(1.0));

    if(!msg1 || !msg2)
    {
      if(!msg1)  std::cerr<<"Failed to find in cache1: "<<msg1<<std::endl;
      if(!msg2)  std::cerr<<"Failed to find in cache2: "<<msg2<<std::endl;
      ros::Duration(0.05).sleep();
    }
    else
    {
      auto frame1= ConvImgMsg(msg1, encoding1);
      auto frame2= ConvImgMsg(msg2, encoding2);
      std::cout<<"frame1.stamp: "<<frame1.Header.stamp<<", frame2.stamp"<<frame2.Header.stamp<<std::endl;

      if(!frame1.Frame.empty())  cv::imshow("camera1", frame1.Frame*255.0*0.3);
      if(!frame2.Frame.empty())  cv::imshow("camera2", frame2.Frame);
    }

    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q')  ros::shutdown();
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
