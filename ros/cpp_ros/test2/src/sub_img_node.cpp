//-------------------------------------------------------------------------------------------
/*! \file    sub_img_node.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.16, 2017

NOTE: Run to activate a camera:
$ rosrun baxter_tools camera_control.py -o left_hand_camera -r 640x400

Or,
$ rosrun usb_cam usb_cam_node

Then run this program:
$ bin/sub_img_node /usb_cam/image_raw

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
// #include <camera_info_manager/camera_info_manager.h>
//-------------------------------------------------------------------------------------------
// namespace trick
// {
// }
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
// using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
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
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::cerr<<"debug.p1"<<std::endl;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  std::cerr<<"cv_ptr: "<<cv_ptr<<std::endl;
  // std::cerr<<"cv_ptr->image: "<<cv_ptr->image<<std::endl;
  cv::Mat frame= cv_ptr->image;

  cv::imshow("camera",frame);
  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  ros::shutdown();
}

int main(int argc, char**argv)
{
  ros::init(argc, argv, "sub_img_node");
  ros::NodeHandle node("~");
  std::string img_topic("/camera/color/image_raw");

  if(argc>1)  img_topic= argv[1];

  cv::namedWindow("camera",1);

  ros::Subscriber sub_img= node.subscribe(img_topic, 1, &ImageCallback);

  ros::spin();

  return 0;
}
//-------------------------------------------------------------------------------------------
