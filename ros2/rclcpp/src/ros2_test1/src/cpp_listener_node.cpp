//-------------------------------------------------------------------------------------------
/*! \file    cpp_listener_node.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
#include <ros2_test1/cpp_listener_node.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------
  // class ListenerNode
  //-------------------------------------------------------------------------------------------
  ListenerNode::ListenerNode(const rclcpp::NodeOptions & options)
  : Node("cpp_listener_node", options)
  {
    sub_ = this->create_subscription<ros2_test1_msgs::msg::MsgTest1>(
      "topic_test", 10,
      std::bind(&ListenerNode::msg_callback, this, std::placeholders::_1));
  }

  void ListenerNode::msg_callback(const ros2_test1_msgs::msg::MsgTest1::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received: [%s], count: %d", msg->status_message.c_str(), msg->count);
  }


//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
