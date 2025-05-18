//-------------------------------------------------------------------------------------------
/*! \file    ros_capture2.cpp
    \brief   Capture example from two ROS image topics (e.g. depth and rgb);
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.15, 2021

$ g++ -O2 -g -W -Wall -o ros_capture2.out ros_capture2.cpp  -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib -I/usr/include/opencv4

$ ./ros_capture2.out

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

typedef void(*TCVCallback)(const cv::Mat&, const cv::Mat&);

cv::Mat frame1, frame2;

void ImageCallback1(const sensor_msgs::ImageConstPtr& msg, const std::string &encoding, TCVCallback callback, bool debug)
{
  if(debug)
  {
    std::cerr<<"msg->header: "<<msg->header;
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
  frame1= cv_ptr->image;
  callback(frame1, frame2);
}
//-------------------------------------------------------------------------------------------

void ImageCallback2(const sensor_msgs::ImageConstPtr& msg, const std::string &encoding, bool debug)
{
  if(debug)
  {
    std::cerr<<"msg->header: "<<msg->header;
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
  frame2= cv_ptr->image;
  // callback(frame1, frame2);
}
//-------------------------------------------------------------------------------------------

void FinishLoop()
{
  ros::shutdown();
}
//-------------------------------------------------------------------------------------------

void StartLoop(int argc, char**argv, const std::string &img1_topic, const std::string &img2_topic, std::string encoding1, std::string encoding2, TCVCallback callback, const std::string &node_name="img_node")
{
  if(node_name!="")  ros::init(argc, argv, node_name);
  ros::NodeHandle node("~");
  if(encoding1=="")  encoding1= GetImageEncoding(img1_topic, node, /*convert_cv=*/true);
  if(encoding2=="")  encoding2= GetImageEncoding(img2_topic, node, /*convert_cv=*/true);

  ros::Subscriber sub_img1= node.subscribe<sensor_msgs::Image>(img1_topic, 1, boost::bind(&ImageCallback1,_1,encoding1,callback,false));
  ros::Subscriber sub_img2= node.subscribe<sensor_msgs::Image>(img2_topic, 1, boost::bind(&ImageCallback2,_1,encoding2,false));

  ros::spin();
}
//-------------------------------------------------------------------------------------------


#ifndef LIBRARY
void CVCallback(const cv::Mat &frame1, const cv::Mat &frame2)
{
  if(!frame1.empty())  cv::imshow("camera1", frame1*255.0*0.3);
  if(!frame2.empty())  cv::imshow("camera2", frame2);
  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img1_topic("/camera/aligned_depth_to_color/image_raw");
  std::string img2_topic("/camera/color/image_raw");
  std::string encoding1(""/*sensor_msgs::image_encodings::TYPE_16UC1*/);
  std::string encoding2(""/*sensor_msgs::image_encodings::BGR8*/);
  if(argc>1)  img1_topic= argv[1];
  if(argc>2)  img2_topic= argv[2];
  if(argc>3)  encoding1= argv[3];
  if(argc>4)  encoding2= argv[4];
  cv::namedWindow("camera1",1);
  cv::namedWindow("camera2",1);
  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback);
  return 0;
}
//-------------------------------------------------------------------------------------------
#endif//LIBRARY
