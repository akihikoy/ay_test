//-------------------------------------------------------------------------------------------
/*! \file    cpp_srv_client.hpp
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#pragma once
//-------------------------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <ros2_test1_msgs/srv/srv_test1.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  class SrvClientNode : public rclcpp::Node
  {
  public:
    explicit SrvClientNode(const rclcpp::NodeOptions & options);
  private:
    void send_request();
    rclcpp::Client<ros2_test1_msgs::srv::SrvTest1>::SharedPtr client_;
    rclcpp::TimerBase::SharedPtr timer_;
  };

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
