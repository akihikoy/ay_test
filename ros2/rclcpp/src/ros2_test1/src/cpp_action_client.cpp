//-------------------------------------------------------------------------------------------
/*! \file    cpp_action_client.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.31, 2025
*/
//-------------------------------------------------------------------------------------------
#include <ros2_test1/cpp_action_client.hpp>
//-------------------------------------------------------------------------------------------
namespace ros2_test1
{
//-------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------
  // class ActionClientNode
  //-------------------------------------------------------------------------------------------
  ActionClientNode::ActionClientNode(const rclcpp::NodeOptions & options)
  : Node("cpp_actionc_node", options)
  {
    client_ = rclcpp_action::create_client<ActionTest1>(this, "action_test");
    timer_ = this->create_wall_timer(std::chrono::milliseconds(500), std::bind(&ActionClientNode::send_goal, this));
  }

  void ActionClientNode::send_goal()
  {
    timer_->cancel();
    if (!client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available");
      return;
    }

    auto goal_msg = ActionTest1::Goal();
    goal_msg.target_count = 5;

    RCLCPP_INFO(this->get_logger(), "Sending goal; %d", goal_msg.target_count);

    auto send_goal_options = rclcpp_action::Client<ActionTest1>::SendGoalOptions();
    send_goal_options.goal_response_callback = std::bind(
      &ActionClientNode::goal_response_callback, this, std::placeholders::_1);
    send_goal_options.feedback_callback = std::bind(
      &ActionClientNode::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback = std::bind(
      &ActionClientNode::result_callback, this, std::placeholders::_1);

    client_->async_send_goal(goal_msg, send_goal_options);
  }

  void ActionClientNode::goal_response_callback(const GoalHandleAction::SharedPtr & goal_handle)
  {
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    } else {
      RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
  }

  void ActionClientNode::feedback_callback(
    GoalHandleAction::SharedPtr,
    const std::shared_ptr<const ActionTest1::Feedback> feedback)
  {
    RCLCPP_INFO(this->get_logger(), "Feedback received: %d", feedback->current_count);
  }

  void ActionClientNode::result_callback(const GoalHandleAction::WrappedResult & result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Result received: %d", result.result->final_count);
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR(this->get_logger(), "Unknown result code: %d", static_cast<int>(result.code));
        break;
    }
  }

//-------------------------------------------------------------------------------------------
}  // end of ros2_test1
//-------------------------------------------------------------------------------------------
