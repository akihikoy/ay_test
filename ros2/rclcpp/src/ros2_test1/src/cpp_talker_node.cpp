//-------------------------------------------------------------------------------------------
/*! \file    cpp_talker_node.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#include "ros2_test1/cpp_talker_node.hpp"
//-------------------------------------------------------------------------------------------
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------
  // class TalkerNode
  //-------------------------------------------------------------------------------------------
  TalkerNode::TalkerNode(const rclcpp::NodeOptions &options)
  : rclcpp::Node("cpp_talker_node", options), count_(0)
  {
    // Parameter
    this->declare_parameter("publish_rate", 1);
    int rate = this->get_parameter("publish_rate").as_int();

    // Publisher
    pub_ = this->create_publisher<ros2_test1_msgs::msg::MsgTest1>("topic_test", /*qos_history_depth=*/10);
    auto period = std::chrono::milliseconds(static_cast<int>(1000.0 / rate));
    timer_ = this->create_wall_timer(period, std::bind(&TalkerNode::on_timer, this));

    RCLCPP_INFO(this->get_logger(), "cpp_talker_node: Initialized");
  }

  void TalkerNode::on_timer()
  {
    auto msg = ros2_test1_msgs::msg::MsgTest1();
    msg.count = count_++;
    msg.status_message = "Counting...";
    pub_->publish(msg);

    RCLCPP_INFO(this->get_logger(), "cpp_talker_node: on_time: %d", msg.count);
  }

  /*
  private:
    // Publisher
    void on_timer();
    rclcpp::Publisher<ros2_test1_msgs::msg::MsgTest1>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int32_t count_;
  */

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
