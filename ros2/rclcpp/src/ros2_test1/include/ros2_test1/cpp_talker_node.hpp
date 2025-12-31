//-------------------------------------------------------------------------------------------
/*! \file    cpp_talker_node.hpp
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#pragma once
//-------------------------------------------------------------------------------------------
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <ros2_test1_msgs/msg/msg_test1.hpp>
#include <ros2_test1_msgs/srv/srv_test1.hpp>
#include <ros2_test1_msgs/action/action_test1.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  class TalkerNode : public rclcpp::Node
  {
  public:
    explicit TalkerNode(const rclcpp::NodeOptions &options);

  private:
    // Publisher
    void on_timer();
    rclcpp::Publisher<ros2_test1_msgs::msg::MsgTest1>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int32_t count_;

  };

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
