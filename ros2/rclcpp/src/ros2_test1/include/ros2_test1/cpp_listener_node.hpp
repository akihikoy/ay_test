//-------------------------------------------------------------------------------------------
/*! \file    cpp_listener_node.hpp
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#pragma once
//-------------------------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <ros2_test1_msgs/msg/msg_test1.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  class ListenerNode : public rclcpp::Node
  {
  public:
    explicit ListenerNode(const rclcpp::NodeOptions & options);
  private:
    void msg_callback(const ros2_test1_msgs::msg::MsgTest1::SharedPtr msg);
    rclcpp::Subscription<ros2_test1_msgs::msg::MsgTest1>::SharedPtr sub_;
  };

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
