//-------------------------------------------------------------------------------------------
/*! \file    cpp_srv_client.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#include <ros2_test1/cpp_srv_client.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------
  // class SrvClientNode
  //-------------------------------------------------------------------------------------------
  SrvClientNode::SrvClientNode(const rclcpp::NodeOptions & options)
  : Node("cpp_srvc_node", options)
  {
    this->declare_parameter("target_value", 100);
    client_ = this->create_client<ros2_test1_msgs::srv::SrvTest1>("srv_test");

    timer_ = this->create_wall_timer(std::chrono::milliseconds(2500), std::bind(&SrvClientNode::send_request, this));
  }

  void SrvClientNode::send_request()
  {
    // timer_->cancel();  // Execute only once.

    // this code blocks the timer callback (not good, in general).
    // if (!client_->wait_for_service(std::chrono::seconds(1))) {
    //   RCLCPP_ERROR(this->get_logger(), "Service not available");
    //   return;
    // }
    // instead, check if the service is available.
    if (!client_->service_is_ready())
    {
      RCLCPP_WARN(this->get_logger(), "Service is not ready yet. Skipping..");
      return;
    }

    auto req = std::make_shared<ros2_test1_msgs::srv::SrvTest1::Request>();
    req->target_value = this->get_parameter("target_value").as_int();

    // asynchronous service call (NOTE: future.get() holds the spin).
    using ServiceFuture = rclcpp::Client<ros2_test1_msgs::srv::SrvTest1>::SharedFuture;
    auto response_received_callback = [this](ServiceFuture future){
      auto result = future.get();
      RCLCPP_INFO(this->get_logger(), "Service Result: %s", result->message.c_str());
    };
    client_->async_send_request(req, response_received_callback);
  }

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
